"""
Generate dense-view plane-refine inputs with GeometryCrafter sequence inference.
"""
import os
import sys
import shutil
from argparse import ArgumentParser
from pathlib import Path

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), '2d-gaussian-splatting'))

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch

from arguments import ModelParams, PipelineParams, get_combined_args
from matcha.dm_scene.cameras import CamerasWrapper, GSCamera
from matcha.pointmap.depthanythingv2 import depth_linear_align
from matcha.pointmap.geometrycrafter import (
    _resize_geometry_payload,
    parse_view_order,
    run_geometrycrafter_sidecar_from_sfm_data,
)
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.point_utils import depth_to_normal
from utils.render_utils import save_img_f32, save_img_u8


def get_surf_cam_normal(view, depth):
    world_normal_map = depth_to_normal(view, depth)
    surf_normal = world_normal_map.permute(2, 0, 1)
    surf_normal_cam = (surf_normal.permute(1, 2, 0) @ (view.world_view_transform[:3, :3])).permute(2, 0, 1)
    return surf_normal, surf_normal_cam


def point_map_to_depth(point_map: np.ndarray) -> np.ndarray:
    depth = point_map[..., 2].astype(np.float32, copy=False)
    depth[~np.isfinite(depth)] = 0.0
    depth[depth < 0.0] = 0.0
    return depth


def ensure_geometrycrafter_rgb_aliases(image_dir: Path, train_viewpoints) -> None:
    for idx, camera in enumerate(train_viewpoints):
        source_path = image_dir / f"rgb_frame{idx:06d}.png"
        alias_path = image_dir / f"{camera.image_name.split('.')[0]}.png"
        if alias_path.exists():
            continue
        if not source_path.exists():
            raise FileNotFoundError(f"Missing dense rgb frame for alias creation: {source_path}")
        try:
            os.symlink(source_path.name, alias_path)
        except OSError:
            shutil.copy(source_path, alias_path)


def scene_camera_to_gs_camera(scene_camera, *, data_device="cpu"):
    return GSCamera(
        colmap_id=scene_camera.colmap_id,
        R=scene_camera.R,
        T=scene_camera.T,
        FoVx=scene_camera.FoVx,
        FoVy=scene_camera.FoVy,
        image=scene_camera.original_image,
        gt_alpha_mask=scene_camera.gt_alpha_mask,
        image_name=scene_camera.image_name,
        uid=scene_camera.uid,
        data_device=data_device,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", required=True, type=str)
    parser.add_argument("--geometrycrafter_repo", type=str, default="/home/rais/GeometryCrafter")
    parser.add_argument("--geometrycrafter_cache_root", type=str, default=None)
    parser.add_argument("--geometrycrafter_num_views", type=int, default=12)
    parser.add_argument("--geometrycrafter_view_order", type=str, default="0,1,10,11,2,3,4,5,6,7,8,9")
    parser.add_argument("--geometrycrafter_model_type", type=str, default="diff", choices=["diff", "determ"])
    parser.add_argument("--geometrycrafter_height", type=int, default=576)
    parser.add_argument("--geometrycrafter_width", type=int, default=1024)
    parser.add_argument("--geometrycrafter_downsample_ratio", type=float, default=1.0)
    parser.add_argument("--geometrycrafter_num_inference_steps", type=int, default=5)
    parser.add_argument("--geometrycrafter_guidance_scale", type=float, default=1.0)
    parser.add_argument("--geometrycrafter_window_size", type=int, default=110)
    parser.add_argument("--geometrycrafter_decode_chunk_size", type=int, default=8)
    parser.add_argument("--geometrycrafter_overlap", type=int, default=25)
    parser.add_argument("--geometrycrafter_process_length", type=int, default=-1)
    parser.add_argument("--geometrycrafter_process_stride", type=int, default=1)
    parser.add_argument("--geometrycrafter_seed", type=int, default=42)
    parser.add_argument("--geometrycrafter_parallel_sequences", type=int, default=1)
    parser.add_argument("--geometrycrafter_force_projection", action="store_true", default=True)
    parser.add_argument("--geometrycrafter_no_force_projection", action="store_false", dest="geometrycrafter_force_projection")
    parser.add_argument("--geometrycrafter_force_fixed_focal", action="store_true", default=True)
    parser.add_argument("--geometrycrafter_no_force_fixed_focal", action="store_false", dest="geometrycrafter_force_fixed_focal")
    parser.add_argument("--geometrycrafter_use_extract_interp", action="store_true")
    parser.add_argument("--geometrycrafter_track_time", action="store_true")
    parser.add_argument("--geometrycrafter_low_memory_usage", action="store_true")
    args = get_combined_args(parser)

    safe_state(False)

    dataset, iteration, pipe = model.extract(args), args.iteration, pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    train_viewpoints = scene.getTrainCameras().copy()
    test_viewpoints = scene.getTestCameras().copy()
    training_gs_cameras = [scene_camera_to_gs_camera(camera, data_device="cpu") for camera in train_viewpoints]
    test_gs_cameras = [scene_camera_to_gs_camera(camera, data_device="cpu") for camera in test_viewpoints]
    training_cameras = CamerasWrapper(training_gs_cameras, no_p3d_cameras=False)
    test_cameras = CamerasWrapper(test_gs_cameras, no_p3d_cameras=False) if len(test_gs_cameras) > 0 else None
    n_views = len(train_viewpoints)

    cur_root_dir = os.path.join(args.source_path, "render-dense-train-views")
    warp_root_dir = os.path.join(cur_root_dir, "dense-render")
    inpaint_root_dir = os.path.join(cur_root_dir, "dense-gt-rgb")
    save_root_dir = os.path.join(args.source_path, "plane-refine-depths")
    if os.path.exists(save_root_dir):
        chart_plane_refine_depth_path = os.path.join(args.source_path, "plane-refine-depths_charts_views")
        cmd = f"mv {save_root_dir} {chart_plane_refine_depth_path}"
        print(cmd)
        os.system(cmd)
    os.makedirs(save_root_dir, exist_ok=True)

    for i in range(n_views):
        inpaint_img_path = os.path.join(inpaint_root_dir, f"rgb_frame{i:06d}.png")
        save_img_path = os.path.join(save_root_dir, f"rgb_frame{i:06d}.png")
        shutil.copy(inpaint_img_path, save_img_path)

    geometrycrafter_args = {
        "height": args.geometrycrafter_height,
        "width": args.geometrycrafter_width,
        "downsample_ratio": args.geometrycrafter_downsample_ratio,
        "num_inference_steps": args.geometrycrafter_num_inference_steps,
        "guidance_scale": args.geometrycrafter_guidance_scale,
        "window_size": args.geometrycrafter_window_size,
        "decode_chunk_size": args.geometrycrafter_decode_chunk_size,
        "overlap": args.geometrycrafter_overlap,
        "process_length": args.geometrycrafter_process_length,
        "process_stride": args.geometrycrafter_process_stride,
        "seed": args.geometrycrafter_seed,
        "model_type": args.geometrycrafter_model_type,
        "force_projection": args.geometrycrafter_force_projection,
        "force_fixed_focal": args.geometrycrafter_force_fixed_focal,
        "use_extract_interp": args.geometrycrafter_use_extract_interp,
        "track_time": args.geometrycrafter_track_time,
        "low_memory_usage": args.geometrycrafter_low_memory_usage,
    }

    cache_root = Path(
        getattr(args, "geometrycrafter_cache_root", None)
        or Path(args.source_path) / "geometrycrafter_dense_sidecar"
    )
    image_dir = Path(inpaint_root_dir)
    ensure_geometrycrafter_rgb_aliases(image_dir, train_viewpoints)
    dense_indices = list(range(n_views))

    scene_pm, _ = run_geometrycrafter_sidecar_from_sfm_data(
        pointmap_cameras=training_cameras,
        training_cameras=training_cameras,
        test_cameras=test_cameras,
        image_dir=image_dir,
        image_indices=dense_indices,
        cache_root=cache_root,
        geometrycrafter_root=args.geometrycrafter_repo,
        view_order=parse_view_order(args.geometrycrafter_view_order, num_views=args.geometrycrafter_num_views),
        num_views=args.geometrycrafter_num_views,
        parallel_sequences=args.geometrycrafter_parallel_sequences,
        device="cpu",
        geometrycrafter_args=geometrycrafter_args,
    )

    visible_threshold = 0.9
    all_points = []
    for i in range(n_views):
        target_height = int(train_viewpoints[i].image_height)
        target_width = int(train_viewpoints[i].image_width)
        confidence_mask = scene_pm.confidence[i].numpy() > 0.5
        camera_space_points, confidence_mask = _resize_geometry_payload(
            scene_pm.points3d[i].numpy(),
            confidence_mask,
            target_height=target_height,
            target_width=target_width,
        )
        mono_depth = point_map_to_depth(camera_space_points)
        mono_depth_tiff_path = os.path.join(save_root_dir, f"mono_depth_frame{i:06d}.tiff")
        save_img_f32(mono_depth, mono_depth_tiff_path)
        plt.imsave(os.path.join(save_root_dir, f"mono_depth_frame{i:06d}.png"), mono_depth, cmap="viridis")

        warp_depth_path = os.path.join(warp_root_dir, f"depth_frame{i:06d}.tiff")
        warp_depth = np.array(Image.open(warp_depth_path)).astype(np.float32)
        warp_depth_t = torch.from_numpy(warp_depth).to("cuda")

        alpha_path = os.path.join(warp_root_dir, f"alpha_{i:06d}.npy")
        alpha = torch.from_numpy(np.load(alpha_path)).to("cuda")
        visible_mask = alpha > visible_threshold
        geometrycrafter_visible_mask = torch.from_numpy(confidence_mask).to("cuda")
        final_visible_mask = visible_mask & geometrycrafter_visible_mask
        np.save(os.path.join(save_root_dir, f"visibility_frame{i:06d}.npy"), final_visible_mask.cpu().numpy())
        Image.fromarray((final_visible_mask.cpu().numpy() * 255.0).astype(np.uint8)).save(
            os.path.join(save_root_dir, f"visibility_frame{i:06d}.png")
        )

        mono_depth_t = torch.from_numpy(np.clip(mono_depth, 1e-6, None)).to("cuda")
        mono_disp = 1.0 / mono_depth_t
        aligned_depth = depth_linear_align(disp=mono_disp, render_depth=warp_depth_t, visible_mask=final_visible_mask)
        surf_normal, surf_normal_cam = get_surf_cam_normal(train_viewpoints[i], aligned_depth.unsqueeze(0))
        surf_normal = surf_normal.permute(1, 2, 0)
        surf_normal_cam = surf_normal_cam.permute(1, 2, 0)

        np.save(os.path.join(save_root_dir, f"depth_normal_world_frame{i:06d}.npy"), surf_normal.cpu().numpy())
        save_img_u8(surf_normal.cpu().numpy() * 0.5 + 0.5, os.path.join(save_root_dir, f"depth_normal_world_frame{i:06d}.png"))

        np.save(os.path.join(save_root_dir, f"mono_normal_world_frame{i:06d}.npy"), surf_normal.cpu().numpy())
        save_img_u8(surf_normal.cpu().numpy() * 0.5 + 0.5, os.path.join(save_root_dir, f"mono_normal_world_frame{i:06d}.png"))

        np.save(os.path.join(save_root_dir, f"mono_normal_frame{i:06d}.npy"), surf_normal_cam.cpu().numpy())
        save_img_u8(surf_normal_cam.cpu().numpy() * 0.5 + 0.5, os.path.join(save_root_dir, f"mono_normal_frame{i:06d}.png"))

        merge_depth = aligned_depth.clone()
        merge_depth[final_visible_mask] = warp_depth_t[final_visible_mask]
        save_img_f32(merge_depth.cpu().numpy(), os.path.join(save_root_dir, f"depth_frame{i:06d}.tiff"))
        plt.imsave(os.path.join(save_root_dir, f"depth_frame{i:06d}.png"), merge_depth.cpu().numpy(), cmap="viridis")

        valid_points = torch.from_numpy(camera_space_points.reshape(-1, 3))
        valid_mask = torch.isfinite(valid_points).all(dim=1) & (valid_points[:, 2] > 0) & torch.from_numpy(confidence_mask.reshape(-1))
        valid_points = valid_points[valid_mask]
        if valid_points.numel() > 0:
            pose = torch.inverse(train_viewpoints[i].world_view_transform.T).cpu().float()
            world_points = valid_points @ pose[:3, :3].T + pose[:3, 3]
            all_points.append(world_points)

        print(f"frame {i:06d} done!")

    if all_points:
        import trimesh

        dense_visible_points = torch.cat(all_points, dim=0)
        dense_visible_points_path = os.path.join(save_root_dir, "dense-view-points.ply")
        trimesh.PointCloud(dense_visible_points.numpy()).export(dense_visible_points_path)
        print(f"Saved dense view points to {dense_visible_points_path}")

    print("All frames done!")
