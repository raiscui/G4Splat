from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

try:
    from sam2.build_sam import build_sam2_video_predictor, build_sam2_video_predictor_hf
except ImportError:  # pragma: no cover - handled at runtime for missing dependency
    build_sam2_video_predictor = None
    build_sam2_video_predictor_hf = None


SAM2_CONFIG = {
    "checkpoint": "./checkpoint/sam2/sam2.1_hiera_large.pt",
    "model_cfg": "configs/sam2.1/sam2.1_hiera_l.yaml",
    "model_id": "facebook/sam2.1-hiera-large",
    "coverage_threshold": 0.65,
    "max_refine_passes": 2,
    "offload_video_to_cpu": False,
    "offload_state_to_cpu": False,
    "async_loading_frames": False,
    "vos_optimized": False,
}


@dataclass
class SAM2MaskGenerator:
    predictor: object
    device: str
    coverage_threshold: float
    max_refine_passes: int
    offload_video_to_cpu: bool
    offload_state_to_cpu: bool
    async_loading_frames: bool

    def _inference_context(self):
        autocast_context = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if str(self.device).startswith("cuda")
            else nullcontext()
        )
        return torch.inference_mode(), autocast_context

    def _init_state(self, video_path: str | Path):
        video_path = str(video_path)
        try:
            return self.predictor.init_state(
                video_path=video_path,
                offload_video_to_cpu=self.offload_video_to_cpu,
                offload_state_to_cpu=self.offload_state_to_cpu,
                async_loading_frames=self.async_loading_frames,
            )
        except TypeError:
            return self.predictor.init_state(video_path)

    def _add_new_mask(self, state, frame_idx: int, obj_id: int, mask: np.ndarray):
        mask = np.asarray(mask, dtype=np.uint8)
        inference_mode, autocast_context = self._inference_context()
        with inference_mode, autocast_context:
            try:
                return self.predictor.add_new_mask(
                    inference_state=state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    mask=mask,
                )
            except TypeError:
                return self.predictor.add_new_mask(state, frame_idx, obj_id, mask)

    def _propagate(self, state) -> dict[int, dict[int, np.ndarray]]:
        tracked_masks: dict[int, dict[int, np.ndarray]] = {}
        inference_mode, autocast_context = self._inference_context()
        with inference_mode, autocast_context:
            for frame_idx, object_ids, mask_logits in self.predictor.propagate_in_video(state):
                mask_array = mask_logits.detach().float().cpu().numpy()
                if mask_array.ndim == 4:
                    mask_array = mask_array[:, 0]
                elif mask_array.ndim == 2:
                    mask_array = mask_array[None, ...]

                frame_masks: dict[int, np.ndarray] = {}
                for i, obj_id in enumerate(object_ids):
                    frame_masks[int(obj_id)] = mask_array[i] > 0.0
                tracked_masks[int(frame_idx)] = frame_masks
        return tracked_masks

    def track_sequence(
        self,
        sequence_dir: str | Path,
        seed_masks_by_frame: Mapping[int, Sequence[np.ndarray]],
    ) -> dict[int, dict[int, np.ndarray]]:
        state = self._init_state(sequence_dir)
        try:
            next_obj_id = 1

            for seed_mask in seed_masks_by_frame.get(0, []):
                if np.asarray(seed_mask, dtype=bool).sum() == 0:
                    continue
                self._add_new_mask(state, frame_idx=0, obj_id=next_obj_id, mask=seed_mask)
                next_obj_id += 1

            tracked_masks = self._propagate(state) if next_obj_id > 1 else {}

            for _ in range(self.max_refine_passes):
                additions = 0
                for frame_idx, candidate_masks in sorted(seed_masks_by_frame.items()):
                    existing_masks = tracked_masks.get(frame_idx, {})
                    for candidate_mask in candidate_masks:
                        candidate_mask = np.asarray(candidate_mask, dtype=bool)
                        candidate_area = int(candidate_mask.sum())
                        if candidate_area == 0:
                            continue

                        best_coverage = 0.0
                        for existing_mask in existing_masks.values():
                            overlap = np.logical_and(existing_mask, candidate_mask).sum()
                            best_coverage = max(best_coverage, overlap / max(candidate_area, 1))

                        if best_coverage >= self.coverage_threshold:
                            continue

                        self._add_new_mask(state, frame_idx=frame_idx, obj_id=next_obj_id, mask=candidate_mask)
                        next_obj_id += 1
                        additions += 1

                if additions == 0:
                    break
                tracked_masks = self._propagate(state)

            return tracked_masks
        finally:
            if hasattr(self.predictor, 'reset_state'):
                self.predictor.reset_state(state)



def setup_sam(
    sam_checkpoint: str = SAM2_CONFIG["checkpoint"],
    model_cfg: str = SAM2_CONFIG["model_cfg"],
    model_id: str = SAM2_CONFIG["model_id"],
    sam2_checkpoint_path: str | None = None,
    sam2_config: str | None = None,
    sam2_model_id: str | None = None,
    device: str = "cuda",
    coverage_threshold: float = SAM2_CONFIG["coverage_threshold"],
    max_refine_passes: int = SAM2_CONFIG["max_refine_passes"],
    offload_video_to_cpu: bool = SAM2_CONFIG["offload_video_to_cpu"],
    offload_state_to_cpu: bool = SAM2_CONFIG["offload_state_to_cpu"],
    async_loading_frames: bool = SAM2_CONFIG["async_loading_frames"],
    vos_optimized: bool = SAM2_CONFIG["vos_optimized"],
):
    if build_sam2_video_predictor is None or build_sam2_video_predictor_hf is None:
        raise ImportError(
            "sam2 is not installed. Install it before running plane_excavator with the SAM2 backend."
        )

    if sam2_checkpoint_path is not None:
        sam_checkpoint = sam2_checkpoint_path
    if sam2_config is not None:
        model_cfg = sam2_config
    if sam2_model_id is not None:
        model_id = sam2_model_id

    checkpoint_path = Path(sam_checkpoint)
    predictor = None

    if checkpoint_path.exists():
        print(f"loading SAM2 checkpoint from: {checkpoint_path}")
        try:
            try:
                predictor = build_sam2_video_predictor(
                    model_cfg,
                    str(checkpoint_path),
                    device=device,
                    vos_optimized=vos_optimized,
                )
            except TypeError:
                predictor = build_sam2_video_predictor(model_cfg, str(checkpoint_path), device=device)
        except Exception as exc:
            print(
                f"Failed to initialize SAM2 from local checkpoint {checkpoint_path}: {exc}. "
                f"Falling back to Hugging Face model: {model_id}"
            )
            predictor = None

    if predictor is None:
        if not checkpoint_path.exists():
            print(
                f"SAM2 checkpoint not found at {checkpoint_path}. Falling back to Hugging Face model: {model_id}"
            )
        predictor = build_sam2_video_predictor_hf(model_id=model_id, device=device, vos_optimized=vos_optimized)

    return SAM2MaskGenerator(
        predictor=predictor,
        device=device,
        coverage_threshold=coverage_threshold,
        max_refine_passes=max_refine_passes,
        offload_video_to_cpu=offload_video_to_cpu,
        offload_state_to_cpu=offload_state_to_cpu,
        async_loading_frames=async_loading_frames,
    )
