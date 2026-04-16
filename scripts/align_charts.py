import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import inspect
import torch
import torch.nn.functional as F
import yaml

from matcha.pointmap.depthanythingv2 import get_pointmap_from_mast3r_scene_with_depthanything
from matcha.dm_scene.cameras import CamerasWrapper, rescale_cameras, create_gs_cameras_from_pointmap
from matcha.dm_trainers.charts_alignment import align_charts_in_parallel

from rich.console import Console


def parse_optional_image_indices(raw_value):
    if raw_value is None:
        return None
    if isinstance(raw_value, list):
        return [int(value) for value in raw_value]

    normalized = str(raw_value).strip()
    if not normalized:
        return None
    if normalized.startswith('[') and normalized.endswith(']'):
        normalized = normalized[1:-1]
    tokens = normalized.replace(',', ' ').split()
    return [int(token) for token in tokens]


def _validate_resolution_factor(raw_value: int) -> int:
    resolution = int(raw_value)
    if resolution <= 0:
        raise ValueError(f"--resolution must be positive, got {raw_value}")
    return resolution


def _downscale_dimension(size: int, resolution: int) -> int:
    if resolution == 1:
        return int(size)
    return max(1, int(round(float(size) / float(resolution))))


def _resize_nhwc_tensor(tensor: torch.Tensor, target_height: int, target_width: int, *, mode: str) -> torch.Tensor:
    if tuple(tensor.shape[1:3]) == (target_height, target_width):
        return tensor
    interpolate_kwargs = {
        "input": tensor.permute(0, 3, 1, 2).float(),
        "size": (target_height, target_width),
        "mode": mode,
    }
    if mode in {"bilinear", "bicubic"}:
        interpolate_kwargs["align_corners"] = False
    resized = F.interpolate(**interpolate_kwargs)
    return resized.permute(0, 2, 3, 1).to(tensor.dtype)


def _resize_nhw_tensor(tensor: torch.Tensor, target_height: int, target_width: int, *, mode: str, threshold: float | None = None) -> torch.Tensor:
    if tuple(tensor.shape[1:3]) == (target_height, target_width):
        return tensor
    interpolate_kwargs = {
        "input": tensor[:, None].float(),
        "size": (target_height, target_width),
        "mode": mode,
    }
    if mode in {"bilinear", "bicubic"}:
        interpolate_kwargs["align_corners"] = False
    resized = F.interpolate(**interpolate_kwargs)[:, 0]
    if threshold is not None:
        return resized > threshold
    return resized.to(tensor.dtype)


def _clone_pointmap_like(scene_pm, **overrides):
    init_kwargs = dict(
        img_paths=list(scene_pm.img_paths),
        images=overrides.get("images", scene_pm.images),
        original_images=overrides.get("original_images", scene_pm.original_images),
        focals=overrides.get("focals", scene_pm.focals),
        poses=overrides.get("poses", scene_pm.poses),
        points3d=overrides.get("points3d", scene_pm.points3d),
        confidence=overrides.get("confidence", scene_pm.confidence),
        masks=overrides.get("masks", scene_pm.masks),
        device=overrides.get("device", scene_pm.device),
    )
    constructor = type(scene_pm)
    signature = inspect.signature(constructor.__init__)
    if "scene_cameras" in signature.parameters and hasattr(scene_pm, "scene_cameras"):
        init_kwargs["scene_cameras"] = scene_pm.scene_cameras
    if "scene_eval_cameras" in signature.parameters and hasattr(scene_pm, "scene_eval_cameras"):
        init_kwargs["scene_eval_cameras"] = scene_pm.scene_eval_cameras
    if "metadata" in signature.parameters and hasattr(scene_pm, "metadata"):
        init_kwargs["metadata"] = dict(scene_pm.metadata)
    return constructor(**init_kwargs)


def _downsample_pointmap_for_alignment(scene_pm, resolution: int):
    resolution = _validate_resolution_factor(resolution)
    if resolution == 1:
        return scene_pm

    target_height = _downscale_dimension(scene_pm.points3d.shape[1], resolution)
    target_width = _downscale_dimension(scene_pm.points3d.shape[2], resolution)
    target_original_height = _downscale_dimension(scene_pm.original_images.shape[1], resolution)
    target_original_width = _downscale_dimension(scene_pm.original_images.shape[2], resolution)

    resized_scene_pm = _clone_pointmap_like(
        scene_pm,
        images=_resize_nhwc_tensor(scene_pm.images, target_height, target_width, mode="bilinear"),
        original_images=_resize_nhwc_tensor(
            scene_pm.original_images,
            target_original_height,
            target_original_width,
            mode="bilinear",
        ),
        focals=scene_pm.focals / float(resolution),
        poses=scene_pm.poses,
        points3d=_resize_nhwc_tensor(scene_pm.points3d, target_height, target_width, mode="bilinear"),
        confidence=_resize_nhw_tensor(scene_pm.confidence, target_height, target_width, mode="bilinear"),
        masks=_resize_nhw_tensor(scene_pm.masks, target_height, target_width, mode="nearest", threshold=0.5),
        device=scene_pm.device,
    )
    return resized_scene_pm


def _resize_depth_maps(depth_maps: torch.Tensor, target_height: int, target_width: int) -> torch.Tensor:
    if tuple(depth_maps.shape[-2:]) == (target_height, target_width):
        return depth_maps
    return F.interpolate(
        depth_maps[:, None],
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    )[:, 0]


def _resize_boolean_maps(boolean_maps: torch.Tensor, target_height: int, target_width: int) -> torch.Tensor:
    if tuple(boolean_maps.shape[-2:]) == (target_height, target_width):
        return boolean_maps
    resized = F.interpolate(
        boolean_maps[:, None].float(),
        size=(target_height, target_width),
        mode="nearest",
    )[:, 0]
    return resized > 0.5


def _prepare_reference_depth_maps(scene_pm, sfm_data, scaled_cameras, scale_factor: float) -> torch.Tensor:
    target_height, target_width = scene_pm.points3d.shape[1:3]
    reference_depth_maps = []
    lowres_cameras = sfm_data['pointmap_cameras']

    for i_chart in range(len(scaled_cameras)):
        image_name = scaled_cameras.gs_cameras[i_chart].image_name.split('.')[0]
        point_ids = sfm_data['image_sfm_points'][image_name]
        reference_depth_values = scaled_cameras.p3d_cameras[i_chart].get_world_to_view_transform().transform_points(
            scale_factor * sfm_data['sfm_xyz'][point_ids]
        )[..., 2]

        source_height = lowres_cameras.gs_cameras[i_chart].image_height
        source_width = lowres_cameras.gs_cameras[i_chart].image_width
        expected_values = source_height * source_width
        if reference_depth_values.numel() != expected_values:
            raise RuntimeError(
                f"Reference depth count for {image_name} is {reference_depth_values.numel()}, "
                f"but pointmap resolution is {source_height}x{source_width} ({expected_values} values). "
                "This path expects dense per-pixel MASt3R pointmaps; keep pointmap.max_sfm_points=null."
            )

        reference_depth_maps.append(reference_depth_values.view(1, source_height, source_width))

    stacked_depth_maps = torch.cat(reference_depth_maps, dim=0)
    return _resize_depth_maps(stacked_depth_maps, target_height, target_width)


def _prepare_geometrycrafter_reference_depth_maps(scene_pm, scaled_cameras, scale_factor: float) -> torch.Tensor:
    target_height, target_width = scene_pm.points3d.shape[1:3]
    reference_depth_maps = []

    for i_chart in range(len(scaled_cameras)):
        points3d = scale_factor * scene_pm.points3d[i_chart].reshape(-1, 3)
        reference_depth_values = scaled_cameras.p3d_cameras[i_chart].get_world_to_view_transform().transform_points(
            points3d
        )[..., 2]
        expected_values = target_height * target_width
        if reference_depth_values.numel() != expected_values:
            raise RuntimeError(
                f"GeometryCrafter reference depth count for chart {i_chart} is {reference_depth_values.numel()}, "
                f"but chart resolution is {target_height}x{target_width} ({expected_values} values)."
            )
        reference_depth_maps.append(reference_depth_values.view(1, target_height, target_width))

    return torch.cat(reference_depth_maps, dim=0)


def _prepare_geometrycrafter_reference_point_clouds(sfm_data, scaled_cameras, scale_factor: float):
    reference_points = []
    for i_chart in range(len(scaled_cameras)):
        image_name = scaled_cameras.gs_cameras[i_chart].image_name.split('.')[0]
        point_ids = sfm_data['image_sfm_points'][image_name]
        if len(point_ids) == 0:
            raise RuntimeError(f"No COLMAP sparse points available for chart {i_chart} ({image_name}).")
        reference_points.append(scale_factor * sfm_data['sfm_xyz'][point_ids])
    return reference_points


def _prepare_alignment_masks(
    *,
    scene_pm,
    mast3r_pm,
    masking_config,
    use_geometrycrafter_masks: bool,
) -> torch.Tensor | None:
    if not masking_config['use_masks_for_alignment']:
        return None

    target_height, target_width = scene_pm.points3d.shape[1:3]
    if use_geometrycrafter_masks:
        geometrycrafter_masks = scene_pm.masks
        geometrycrafter_masks = _resize_boolean_maps(
            geometrycrafter_masks,
            target_height=target_height,
            target_width=target_width,
        )
        return geometrycrafter_masks

    if mast3r_pm is None:
        raise ValueError("mast3r_pm is required when using MASt3R-based alignment masks.")

    mast3r_masks = mast3r_pm.confidence > masking_config['sfm_mask_threshold']
    mast3r_masks = _resize_boolean_maps(
        mast3r_masks,
        target_height=target_height,
        target_width=target_width,
    )
    return mast3r_masks


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Scene arguments
    parser.add_argument('-s', '--source_path', type=str, required=True)
    parser.add_argument('-m', '--mast3r_scene', type=str, required=True)
    parser.add_argument('--colmap_scene', type=str, default=None)
    parser.add_argument('-o', '--output_path', type=str, default=None)
    parser.add_argument('--depth_model', type=str, default="depthanythingv2")
    parser.add_argument(
        '--geometry_prior_mode',
        choices=['baseline', 'sidecar-only', 'hybrid-override-at-align-prep'],
        default='baseline',
        help='Stage-1 experiment mode. baseline keeps the current DepthAnythingV2 path; sidecar-only only materializes GeometryCrafter priors; hybrid-override-at-align-prep reuses the baseline backbone and overrides only the per-frame geometry payload in align_charts preparation.',
    )
    parser.add_argument('--white_background', type=bool, default=False)
    
    # DepthAnything arguments
    parser.add_argument('--depthanythingv2_checkpoint_dir', type=str, default='./Depth-Anything-V2/checkpoints/')
    parser.add_argument('--depthanything_encoder', type=str, default='vitl')

    # GeometryCrafter arguments
    parser.add_argument('--geometrycrafter_repo', type=str, default='/home/rais/GeometryCrafter')
    parser.add_argument('--geometrycrafter_cache_root', type=str, default=None)
    parser.add_argument('--geometrycrafter_num_views', type=int, default=12)
    parser.add_argument(
        '--geometrycrafter_view_order',
        type=str,
        default='0,1,10,11,2,3,4,5,6,7,8,9',
        help='Comma-separated source-view ids matching the interleaved COLMAP order.',
    )
    parser.add_argument('--geometrycrafter_model_type', type=str, default='diff', choices=['diff', 'determ'])
    parser.add_argument(
        '--geometrycrafter_height',
        type=int,
        default=576,
        help='GeometryCrafter processing height. Defaults to the recommended 576.',
    )
    parser.add_argument(
        '--geometrycrafter_width',
        type=int,
        default=1024,
        help='GeometryCrafter processing width. Defaults to the recommended 1024.',
    )
    parser.add_argument('--geometrycrafter_downsample_ratio', type=float, default=1.0)
    parser.add_argument('--geometrycrafter_num_inference_steps', type=int, default=5)
    parser.add_argument('--geometrycrafter_guidance_scale', type=float, default=1.0)
    parser.add_argument('--geometrycrafter_window_size', type=int, default=110)
    parser.add_argument('--geometrycrafter_decode_chunk_size', type=int, default=8)
    parser.add_argument('--geometrycrafter_overlap', type=int, default=25)
    parser.add_argument('--geometrycrafter_process_length', type=int, default=-1)
    parser.add_argument('--geometrycrafter_process_stride', type=int, default=1)
    parser.add_argument('--geometrycrafter_seed', type=int, default=42)
    parser.add_argument('--geometrycrafter_force_projection', action='store_true', default=True)
    parser.add_argument('--geometrycrafter_no_force_projection', action='store_false', dest='geometrycrafter_force_projection')
    parser.add_argument('--geometrycrafter_force_fixed_focal', action='store_true', default=True)
    parser.add_argument('--geometrycrafter_no_force_fixed_focal', action='store_false', dest='geometrycrafter_force_fixed_focal')
    parser.add_argument('--geometrycrafter_use_extract_interp', action='store_true')
    parser.add_argument('--geometrycrafter_track_time', action='store_true')
    parser.add_argument('--geometrycrafter_low_memory_usage', action='store_true')
    parser.add_argument('--geometrycrafter_parallel_sequences', type=int, default=1,
                        help='How many GeometryCrafter interleaved sequences to execute in parallel.')
    
    # Deprecated arguments (should not be used)
    parser.add_argument('--image_indices', type=str, default=None)
    parser.add_argument('--n_charts', type=int, default=None)
    
    # Config
    parser.add_argument('-c', '--config', type=str, default='default')
    parser.add_argument(
        '--resolution',
        type=int,
        default=1,
        help='Downsampling factor for align_charts inputs. Use 2 for half-resolution, 4 for quarter-resolution.',
    )
    
    args = parser.parse_args()
    args.resolution = _validate_resolution_factor(args.resolution)
    
    # Set console
    CONSOLE = Console(width=120)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set output path
    if args.output_path is None:
        args.output_path = args.mast3r_scene
    else:
        os.makedirs(args.output_path, exist_ok=True)
    CONSOLE.print(f"[INFO] Aligned charts will be saved to: {args.output_path}")
    
    # Load config
    config_path = os.path.join('configs/charts_alignment', args.config + '.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    pm_config = config['pointmap']
    scene_config = config['scene']
    align_config = config['alignment']
    masking_config = config['masking']
    
    # Reprojection loss
    if align_config['use_reprojection_loss']:
        raise NotImplementedError("Reprojection loss is not implemented yet.")

    parsed_image_indices = parse_optional_image_indices(args.image_indices)

    use_geometrycrafter = args.depth_model.lower() == 'geometrycrafter' or args.geometry_prior_mode != 'baseline'
    if args.depth_model.lower() == 'geometrycrafter' and args.geometry_prior_mode == 'baseline':
        CONSOLE.print('[INFO] depth_model=geometrycrafter requested without an explicit mode; using hybrid-override-at-align-prep.')
        args.geometry_prior_mode = 'hybrid-override-at-align-prep'

    if use_geometrycrafter:
        from matcha.pointmap.geometrycrafter import (
            get_pointmap_from_colmap_scene_with_geometrycrafter,
            get_pointmap_from_mast3r_scene_with_geometrycrafter,
            parse_view_order,
        )

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

        geometrycrafter_kwargs = dict(
            n_images_in_pointmap=args.n_charts,
            image_indices=parsed_image_indices,
            white_background=args.white_background,
            geometrycrafter_root=args.geometrycrafter_repo,
            geometrycrafter_cache_root=args.geometrycrafter_cache_root or os.path.join(args.output_path, 'geometrycrafter_cache'),
            geometrycrafter_num_views=args.geometrycrafter_num_views,
            geometrycrafter_view_order=parse_view_order(
                args.geometrycrafter_view_order,
                num_views=args.geometrycrafter_num_views,
            ),
            geometrycrafter_parallel_sequences=args.geometrycrafter_parallel_sequences,
            geometrycrafter_args=geometrycrafter_args,
            sidecar_only=args.geometry_prior_mode == 'sidecar-only',
            device=device,
            return_sfm_data=True,
            **pm_config,
        )
        if args.colmap_scene is not None:
            scene_pm, sfm_data = get_pointmap_from_colmap_scene_with_geometrycrafter(
                colmap_source_path=args.colmap_scene,
                scene_source_path=args.source_path,
                **geometrycrafter_kwargs,
            )
            mast3r_pm = None
        else:
            scene_pm, sfm_data, mast3r_pm = get_pointmap_from_mast3r_scene_with_geometrycrafter(
                mast3r_scene_source_path=args.mast3r_scene,
                return_mast3r_pointmap=True,
                scene_source_path=args.source_path,
                **geometrycrafter_kwargs,
            )
        if args.geometry_prior_mode == 'sidecar-only':
            CONSOLE.print(f"[INFO] GeometryCrafter sidecar outputs prepared at: {scene_pm.metadata.get('cache_root')}")
            CONSOLE.print(f"[INFO] GeometryCrafter manifest: {scene_pm.metadata.get('sidecar_manifest_path')}")
            sys.exit(0)
    else:
        # Build pointmap from MASt3R-SfM data
        scene_pm, sfm_data, mast3r_pm = get_pointmap_from_mast3r_scene_with_depthanything(
            scene_source_path=args.source_path,
            n_images_in_pointmap=args.n_charts,
            image_indices=parsed_image_indices,
            white_background=args.white_background,
            # MASt3R
            mast3r_scene_source_path=args.mast3r_scene,
            # DepthAnything
            depthanything_checkpoint_dir=args.depthanythingv2_checkpoint_dir,
            depthanything_encoder=args.depthanything_encoder,
            # Misc
            device=device,
            return_sfm_data=True,
            return_mast3r_pointmap=True,
            **pm_config,
        )

    if args.resolution != 1:
        original_chart_height, original_chart_width = scene_pm.points3d.shape[1:3]
        scene_pm = _downsample_pointmap_for_alignment(scene_pm, args.resolution)
        CONSOLE.print(
            f"[INFO] align_charts resolution={args.resolution}: "
            f"chart resolution {original_chart_width}x{original_chart_height} -> "
            f"{scene_pm.points3d.shape[2]}x{scene_pm.points3d.shape[1]}"
        )
    
    # Compute rescaling factor
    _cam_list = create_gs_cameras_from_pointmap(
        scene_pm,
        image_resolution=1,
        load_gt_images=True, 
        max_img_size=pm_config['max_img_size'], 
        use_original_image_size=True,
        average_focal_distances=False,
        verbose=False,
    )
    _pointmap_cameras = CamerasWrapper(_cam_list, no_p3d_cameras=False)
    _scale_factor = scene_config['target_scale'] / _pointmap_cameras.get_spatial_extent()
    
    # Rescale cameras
    _pointmap_cameras = rescale_cameras(_pointmap_cameras, _scale_factor)
    
    # Rescale and prepare reference data based on SFM method
    use_geometrycrafter_alignment_inputs = use_geometrycrafter and args.colmap_scene is not None
    use_geometrycrafter_sparse_point_reference = use_geometrycrafter_alignment_inputs
    if use_geometrycrafter_sparse_point_reference:
        reference_data = _prepare_geometrycrafter_reference_point_clouds(
            sfm_data,
            _pointmap_cameras,
            _scale_factor,
        )
        if align_config.get('use_matching_loss'):
            CONSOLE.print("[INFO] Disabling matching loss for GeometryCrafter COLMAP sparse-point alignment reference.")
        align_config['use_matching_loss'] = False
    else:
        reference_data = _prepare_reference_depth_maps(
            scene_pm,
            sfm_data,
            _pointmap_cameras,
            _scale_factor,
        )

    alignment_masks = _prepare_alignment_masks(
        scene_pm=scene_pm,
        mast3r_pm=mast3r_pm,
        masking_config=masking_config,
        use_geometrycrafter_masks=use_geometrycrafter_alignment_inputs,
    )
    if alignment_masks is not None:
        CONSOLE.print(f"[INFO] {alignment_masks.sum()} points in masks.")
    else:
        CONSOLE.print("[INFO] All points will be used for charts alignment.")
    
    # Align the charts
    output = align_charts_in_parallel(
        # Scene
        scene_pm,
        # Data parameters
        reference_data,
        masks=alignment_masks,
        rendering_size=pm_config['max_img_size'],
        target_scale=scene_config['target_scale'],
        verbose=True,
        return_training_losses=True,
        reprojection_matches_file=None,
        save_charts_data=True,
        charts_data_path=args.output_path,
        **align_config,
    )

    if align_config['use_learnable_confidence']:
        output_verts, output_depths, output_confs, training_losses = output
        output_confs = output_confs - 1.
    else:
        output_verts, output_depths, training_losses = output

    CONSOLE.print("\nInitialization complete!")
    CONSOLE.print("Output vertices shape:", output_verts.shape)
    CONSOLE.print("Output depths shape:", output_depths.shape)
    if align_config['use_learnable_confidence']:
        CONSOLE.print("Output confidence shape:", output_confs.shape)
