from __future__ import annotations

import argparse
import gc
import os
import sys
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans

sys.path.append(os.getcwd())

from mask_generator import setup_sam
from sam2_sequence import DEFAULT_VIEW_ORDER, build_sequences, materialize_sequence_jpgs
from tools import merge_normal_clusters, remove_small_isolated_areas

sys.path.append(os.path.join(os.getcwd(), '2d-gaussian-splatting'))
from utils.general_utils import seed_everything


def normals_cluster(
    normals: np.ndarray,
    img_shape: tuple[int, int],
    n_init_clusters: int = 8,
    n_clusters: int = 6,
    min_size_ratio: float = 0.004,
) -> list[np.ndarray]:
    normals_flat = normals.reshape(-1, 3) if len(normals.shape) == 3 else normals
    valid_mask = np.isfinite(normals_flat).all(axis=1)
    valid_normals = normals_flat[valid_mask]
    if valid_normals.size == 0:
        return []

    effective_init_clusters = min(n_init_clusters, len(valid_normals))
    effective_clusters = min(n_clusters, effective_init_clusters)
    if effective_init_clusters <= 0 or effective_clusters <= 0:
        return []

    kmeans = KMeans(n_clusters=effective_init_clusters, random_state=0, n_init=1).fit(valid_normals)
    pred_valid = kmeans.labels_
    centers = kmeans.cluster_centers_

    count_values = np.bincount(pred_valid)
    topk = np.argpartition(count_values, -effective_clusters)[-effective_clusters:]
    sorted_topk_idx = np.argsort(count_values[topk])
    sorted_topk = topk[sorted_topk_idx][::-1]

    pred_valid, sorted_topk, num_clusters = merge_normal_clusters(pred_valid, sorted_topk, centers)

    min_plane_size = img_shape[0] * img_shape[1] * min_size_ratio
    normal_masks: list[np.ndarray] = []

    for cluster_idx in range(num_clusters):
        mask = np.zeros(normals_flat.shape[0], dtype=bool)
        mask[valid_mask] = pred_valid == sorted_topk[cluster_idx]
        mask_clean = remove_small_isolated_areas(
            (mask > 0).reshape(*img_shape) * 255,
            min_size=min_plane_size,
        ).reshape(-1)
        mask[mask_clean == 0] = 0

        num_labels, labels = cv2.connectedComponents((mask * 255).reshape(img_shape).astype(np.uint8))
        for label in range(1, num_labels):
            normal_masks.append(labels == label)
    return normal_masks


@dataclass
class PlaneExcavatorConfig:
    min_size_ratio: float = 0.01
    n_init_normal_clusters: int = 8
    n_normal_clusters: int = 6
    sam2_checkpoint: str = "./checkpoint/sam2/sam2.1_hiera_large.pt"
    sam2_model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2_model_id: str = "facebook/sam2.1-hiera-large"
    sam2_vos_optimized: bool = False
    sam2_coverage_threshold: float = 0.65
    sam2_max_refine_passes: int = 2
    interleaved_view_count: int = 12
    interleaved_view_order: tuple[int, ...] | None = DEFAULT_VIEW_ORDER
    sequence_layout: str = "auto"
    min_cluster_track_overlap: float = 0.05


class PlaneExcavator:
    def __init__(
        self,
        config: PlaneExcavatorConfig,
        device,
        img_height: int,
        img_width: int,
        use_normal_estimator: bool = False,
        normal_model_type: str = "stablenormal",
    ):
        self.img_shape = (img_height, img_width)
        self.min_plane_size = self.img_shape[0] * self.img_shape[1] * config.min_size_ratio
        self.n_init_normal_clusters = config.n_init_normal_clusters
        self.n_normal_clusters = config.n_normal_clusters
        self.min_cluster_track_overlap = config.min_cluster_track_overlap
        self.device = str(device)
        self.interleaved_view_count = config.interleaved_view_count
        self.interleaved_view_order = config.interleaved_view_order
        self.sequence_layout = config.sequence_layout

        self.sam_model = setup_sam(
            sam_checkpoint=config.sam2_checkpoint,
            model_cfg=config.sam2_model_cfg,
            model_id=config.sam2_model_id,
            device=self.device,
            coverage_threshold=config.sam2_coverage_threshold,
            max_refine_passes=config.sam2_max_refine_passes,
            vos_optimized=config.sam2_vos_optimized,
        )

        self.use_normal_estimator = use_normal_estimator
        if self.use_normal_estimator:
            if normal_model_type != "stablenormal":
                raise ValueError(f"normal_model_type {normal_model_type} is not supported")
            self.normal_estimator = torch.hub.load("Stable-X/StableNormal", "StableNormal", trust_repo=True)

    def _normals_cluster(self, normals: np.ndarray) -> list[np.ndarray]:
        return normals_cluster(
            normals,
            self.img_shape,
            n_init_clusters=self.n_init_normal_clusters,
            n_clusters=self.n_normal_clusters,
            min_size_ratio=self.min_plane_size / (self.img_shape[0] * self.img_shape[1]),
        )

    @staticmethod
    def _load_rgb(rgb_path: str | os.PathLike[str]) -> np.ndarray:
        with Image.open(rgb_path) as image:
            return np.array(image)

    def _load_normals(self, rgb: np.ndarray, normal_path: str | None) -> np.ndarray:
        if self.use_normal_estimator or normal_path is None:
            normals = self.normal_estimator(Image.fromarray(rgb))
            normals = np.array(normals) / 255.0
            return (0.5 - normals) * 2
        return np.load(normal_path)

    def _assign_masks_to_clusters(
        self,
        tracked_masks: dict[int, np.ndarray],
        normal_clusters: list[np.ndarray],
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        plane_mask = np.zeros(self.img_shape, dtype=np.int32)
        plane_instances: dict[int, np.ndarray] = {}

        for cluster_mask in normal_clusters:
            cluster_area = int(cluster_mask.sum())
            if cluster_area < self.min_plane_size:
                continue

            best_obj_id = None
            best_overlap = 0
            for obj_id, tracked_mask in tracked_masks.items():
                overlap = int(np.logical_and(cluster_mask, tracked_mask).sum())
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_obj_id = obj_id

            if best_obj_id is None:
                continue

            overlap_ratio = best_overlap / max(cluster_area, 1)
            if overlap_ratio < self.min_cluster_track_overlap:
                continue

            plane_mask[cluster_mask] = best_obj_id
            if best_obj_id in plane_instances:
                plane_instances[best_obj_id] = np.logical_or(plane_instances[best_obj_id], cluster_mask)
            else:
                plane_instances[best_obj_id] = cluster_mask.copy()

        return plane_mask, [plane_instances[obj_id] for obj_id in sorted(plane_instances)]

    @staticmethod
    def _mask_stats(normals: np.ndarray, plane_instances: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        if not plane_instances:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.int32)

        avg_normals = []
        areas = []
        for mask in plane_instances:
            if mask.sum() == 0:
                continue
            mean_normal = np.mean(normals[mask], axis=0)
            norm = np.linalg.norm(mean_normal)
            if norm > 0:
                mean_normal = mean_normal / norm
            avg_normals.append(mean_normal.astype(np.float32))
            areas.append(int(mask.sum()))
        if not avg_normals:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.int32)
        return np.stack(avg_normals, axis=0), np.asarray(areas, dtype=np.int32)

    def _run_sequence_group(self, sequence_frames):
        rgb_frames = [self._load_rgb(frame.rgb_path) for frame in sequence_frames]
        normal_frames = [self._load_normals(rgb, frame.normal_path) for frame, rgb in zip(sequence_frames, rgb_frames)]
        clusters_per_frame = [self._normals_cluster(normals) for normals in normal_frames]
        seed_masks_by_frame = {idx: clusters for idx, clusters in enumerate(clusters_per_frame)}

        with materialize_sequence_jpgs(sequence_frames) as sequence_dir:
            tracked_masks = self.sam_model.track_sequence(sequence_dir, seed_masks_by_frame)

        outputs = {}
        for local_idx, (frame, rgb, normals, normal_clusters) in enumerate(
            zip(sequence_frames, rgb_frames, normal_frames, clusters_per_frame)
        ):
            plane_mask, plane_instances = self._assign_masks_to_clusters(
                tracked_masks.get(local_idx, {}),
                normal_clusters,
            )
            avg_normals, areas = self._mask_stats(normals, plane_instances)
            outputs[frame.frame_idx] = {
                "seg_mask": plane_mask,
                "normal": avg_normals,
                "areas": areas,
                "vis": {
                    "image": rgb,
                    "pred_norm": np.clip(((normals + 1) * 0.5) * 255, 0, 255).astype(np.uint8),
                    "plane_mask": plane_instances,
                },
            }
        return outputs

    def run_dataset(self, data_path: str):
        grouped_sequences = build_sequences(
            data_path,
            num_views=self.interleaved_view_count,
            view_order=self.interleaved_view_order,
            layout=self.sequence_layout,
        )
        outputs = {}
        for sequence in grouped_sequences.sequences:
            if not sequence:
                continue
            outputs.update(self._run_sequence_group(sequence))
        return outputs


def _save_outputs(data_path: str, outputs: dict[int, dict]):
    from disp import overlay_masks

    for frame_id in sorted(outputs):
        output = outputs[frame_id]
        plane_mask = output["seg_mask"]
        npy_save_path = os.path.join(data_path, f"plane_mask_frame{frame_id:06d}.npy")
        np.save(npy_save_path, plane_mask)
        print(f"save to {npy_save_path}")

        vis_map = overlay_masks(output["vis"]["image"], output["vis"]["plane_mask"])
        save_path = os.path.join(data_path, f"plane_vis_frame{frame_id:06d}.png")
        Image.fromarray(vis_map).save(save_path)
        print(f"save to {save_path}")


def parse_view_order(text: str | None):
    if text is None or text == "":
        return None
    return tuple(int(item.strip()) for item in text.split(",") if item.strip())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plane_root_path", type=str, required=True)
    parser.add_argument("--use_normal_estimator", action="store_true")
    parser.add_argument("--sam2_checkpoint", type=str, default="./checkpoint/sam2/sam2.1_hiera_large.pt")
    parser.add_argument("--sam2_model_cfg", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--sam2_model_id", type=str, default="facebook/sam2.1-hiera-large")
    parser.add_argument("--sam2_vos_optimized", action="store_true")
    parser.add_argument("--sam2_coverage_threshold", type=float, default=0.65)
    parser.add_argument("--sam2_max_refine_passes", type=int, default=2)
    parser.add_argument("--interleaved_view_count", "--num_views", dest="interleaved_view_count", type=int, default=12)
    parser.add_argument("--sequence_layout", choices=["auto", "single", "interleaved"], default="auto")
    parser.add_argument(
        "--interleaved_view_order",
        "--view_order",
        type=str,
        default=None,
        help="Comma-separated source-view ids for each position in an interleaved group, e.g. 0,1,10,11,2,3,4,5,6,7,8,9",
    )
    parser.add_argument("--min_cluster_track_overlap", type=float, default=0.05)
    args = parser.parse_args()

    seed_everything()

    data_path = args.plane_root_path
    rgb_candidates = sorted(file for file in os.listdir(data_path) if file.endswith(".png") and "rgb_frame" in file)
    if not rgb_candidates:
        raise FileNotFoundError(f"No rgb_frame*.png files found in {data_path}")

    with Image.open(os.path.join(data_path, rgb_candidates[0])) as temp_rgb:
        img_height, img_width = temp_rgb.height, temp_rgb.width
    print(f"img_height: {img_height}, img_width: {img_width}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = PlaneExcavatorConfig(
        sam2_checkpoint=args.sam2_checkpoint,
        sam2_model_cfg=args.sam2_model_cfg,
        sam2_model_id=args.sam2_model_id,
        sam2_vos_optimized=args.sam2_vos_optimized,
        sam2_coverage_threshold=args.sam2_coverage_threshold,
        sam2_max_refine_passes=args.sam2_max_refine_passes,
        interleaved_view_count=args.interleaved_view_count,
        interleaved_view_order=parse_view_order(args.interleaved_view_order) or DEFAULT_VIEW_ORDER[: args.interleaved_view_count],
        sequence_layout=args.sequence_layout,
        min_cluster_track_overlap=args.min_cluster_track_overlap,
    )
    plane_excavator = PlaneExcavator(
        config,
        device,
        img_height,
        img_width,
        use_normal_estimator=args.use_normal_estimator,
    )

    if args.use_normal_estimator:
        print("NOTE: use normal estimator for plane extraction")

    print("********** start plane extraction **********")
    outputs = plane_excavator.run_dataset(data_path)
    _save_outputs(data_path, outputs)

    torch.cuda.empty_cache()
    gc.collect()
