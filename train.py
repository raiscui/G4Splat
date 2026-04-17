import os
import sys
import argparse
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import shutil

def run_command_safe(command):
    print(f"Running command: {command}")
    exit_code = os.system(command)
    if exit_code != 0:
        print("Command failed!")
        sys.exit(1)
    else:
        print("Command succeeded!")


def append_optional_arg(command_parts, flag, value):
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            command_parts.append(flag)
        return
    command_parts.extend([flag, str(value)])


def serialize_image_indices(image_indices):
    if image_indices is None:
        return None
    return ",".join(str(int(idx)) for idx in image_indices)


def copy_sparse_point_files(source_model_root, dest_model_root):
    os.makedirs(dest_model_root, exist_ok=True)
    for name in ("points3D.bin", "points3D.txt", "points3D.ply"):
        src_path = os.path.join(source_model_root, name)
        dst_path = os.path.join(dest_model_root, name)
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
        else:
            print(f"[INFO] Skipping missing sparse point file: {src_path}")


def should_skip_initial_chart_plane_refine(depth_model, geometry_prior_mode):
    return geometry_prior_mode == "sidecar-only"


def load_or_create_dense_view_indices(source_path):
    dense_view_json_path = os.path.join(source_path, 'dense_view.json')
    print(f'[INFO]: Search dense view json in {dense_view_json_path}')
    if not os.path.exists(dense_view_json_path):
        source_img_path = os.path.join(source_path, 'images')
        source_img_num = len(os.listdir(source_img_path))
        dense_view_idx_list = list(range(source_img_num))
        print(f"Use all {source_img_num} views in the source path as dense view")
        with open(dense_view_json_path, 'w') as f:
            json.dump({'train': dense_view_idx_list}, f)
        print(f"Save dense view index list to {dense_view_json_path}")
        return dense_view_json_path, dense_view_idx_list

    with open(dense_view_json_path, 'r') as f:
        dense_view_json = json.load(f)
    dense_view_idx_list = [int(idx) for idx in dense_view_json['train']]
    print(f"Use {len(dense_view_idx_list)} views from {dense_view_json_path} as dense view")
    return dense_view_json_path, dense_view_idx_list


def resolve_initial_view_selection(
    *,
    use_view_config,
    source_path,
    config_view_num,
    n_images,
    image_idx,
    use_dense_view,
    dense_view_idx_list,
):
    if use_view_config:
        view_config_path = os.path.join(source_path, f'split-{config_view_num}views.json')
        if os.path.exists(view_config_path):
            with open(view_config_path, 'r') as f:
                view_config = json.load(f)
            image_idx_list = view_config['train']
        else:
            view_config_path = os.path.join(source_path, f'train_test_split_{config_view_num}.json')
            with open(view_config_path, 'r') as f:
                view_config = json.load(f)
            image_idx_list = view_config['train_ids']
        return None, image_idx_list, "view_config"

    if image_idx is not None:
        return None, image_idx, "explicit_image_idx"

    if n_images is not None:
        return n_images, None, "explicit_n_images"

    if use_dense_view and dense_view_idx_list is not None:
        print(f"[INFO] Bootstrapping run_sfm/align_charts from dense_view.json with {len(dense_view_idx_list)} views.")
        return None, dense_view_idx_list, "dense_view_json"

    return None, None, "all_images"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Scene arguments
    parser.add_argument('-s', '--source_path', type=str, required=True, help='Path to the source directory')
    parser.add_argument('-o', '--output_path', type=str, default=None, help='Path to the output directory')
    
    # Image selection parameters
    parser.add_argument('--n_images', type=int, default=None, 
        help='Number of images to use for optimization, sampled with constant spacing. If not provided, all images will be used.')
    parser.add_argument('--use_view_config', action='store_true', 
        help='Use view config file to select images for optimization. If provided, this will override the --n_images and --image_idx arguments.')
    parser.add_argument('--config_view_num', type=int, default=10, 
        help='View number of the config file. If provided, this will override the --n_images.')
    parser.add_argument('--image_idx', type=int, nargs='*', default=None, 
        help='View indices to use for optimization (zero-based indexing). If provided, this will override the --n_images.')
    parser.add_argument('--randomize_images', action='store_true', 
        help='Shuffle training images before sampling with constant spacing. If image_idx is provided, this will be ignored.')
    
    # Dense supervision (Optional)
    parser.add_argument('--dense_supervision', action='store_true', 
        help='Use dense RGB supervision with a COLMAP dataset. Should only be used with --sfm_config posed.')
    parser.add_argument('--dense_regul', type=str, default='default', help='Strength of dense regularization. Can be "default", "strong", "weak", or "none".')
    
    # Output mesh parameters
    parser.add_argument('--use_multires_tsdf', action='store_true', help='Use multi-resolution TSDF fusion instead of adaptive tetrahedralization for mesh extraction (not recommended).')
    parser.add_argument('--no_interpolated_views', action='store_true', help='Disable interpolated views for mesh extraction.')
    
    # SfM config
    parser.add_argument('--sfm_config', type=str, default='unposed', help='Config for SfM. Should be "unposed" or "posed".')
    
    # Chart alignment config
    parser.add_argument('--alignment_config', type=str, default='default', help='Config for charts alignment')
    parser.add_argument('--depth_model', type=str, default="depthanythingv2")
    parser.add_argument(
        '--geometry_prior_mode',
        choices=['baseline', 'sidecar-only', 'hybrid-override-at-align-prep'],
        default='baseline',
        help='Stage-1 geometry prior experiment mode for align_charts.',
    )
    parser.add_argument('--depthanythingv2_checkpoint_dir', type=str, default='./Depth-Anything-V2/checkpoints/')
    parser.add_argument('--depthanything_encoder', type=str, default='vitl')
    parser.add_argument('--geometrycrafter_repo', type=str, default='/home/rais/GeometryCrafter')
    parser.add_argument('--geometrycrafter_cache_root', type=str, default=None)
    parser.add_argument('--geometrycrafter_num_views', type=int, default=12)
    parser.add_argument('--geometrycrafter_view_order', type=str, default='0,1,10,11,2,3,4,5,6,7,8,9')
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
        help='How many GeometryCrafter interleaved sequences to execute in parallel.'
    )
    
    # Free Gaussians config
    parser.add_argument('--free_gaussians_config', type=str, default=None, 
        help='Config for Free Gaussians refinement. '\
        'By default, the config used is "default" for sparse supervision, and "long" for dense supervision.'
    )
    
    # Multi-resolution TSDF config
    parser.add_argument('--tsdf_config', type=str, default='default', help='Config for multi-resolution TSDF fusion')
    
    # Tetrahedralization config
    parser.add_argument('--tetra_config', type=str, default='default', help='Config for adaptive tetrahedralization')
    parser.add_argument('--tetra_downsample_ratio', type=float, default=0.5, 
        help='Downsample ratio for tetrahedralization. We recommend starting with 0.5 and then decreasing to 0.25 '\
        'if the mesh is too dense, or increasing to 1.0 if the mesh is too sparse.'
    )

    # G4Splat config
    parser.add_argument('--resolution', type=int, default=1,
        help='Downsampling factor for image-based stages. Use 1 for original size, 2 for half-resolution, 4 for quarter-resolution.'
    )
    parser.add_argument('--merge_resolution_scale', type=float, default=1.0,
        help='Downscale only the merge_global_3Dplane camera/mask resolution. Use 2.0 for half resolution.'
    )
    parser.add_argument('--merge_device', type=str, default='cpu', choices=['cpu', 'cuda'],
        help='Device for merge_global_3Dplane tensors.'
    )
    parser.add_argument('--select_inpaint_num', type=int, default=20, help='Number of views to select for inpainting.')
    parser.add_argument('--use_downsample_gaussians', action='store_true', help='Use downsample gaussians for training')
    parser.add_argument('--use_mesh_filter', action='store_true', help='Use mesh filter')
    parser.add_argument('--use_dense_view', action='store_true', help='Use dense view for training')                    # Add an additional input stage to extend plane-aware depth estimation across all input views
    parser.add_argument('--skip_render_all_img', action='store_true',
        help='Skip exporting all rendered training images before mesh extraction.'
    )
    parser.add_argument('--export_workers', type=int, default=None,
        help='Parallel workers for render_all_img export, forwarded to render_multires.py.'
    )
    parser.add_argument('--mip_filter_variance', type=float, default=None,
        help='Override mip filter strength during free-gaussians training. Lower keeps more distant detail.'
    )
    parser.add_argument('--checkpoint_iterations', type=int, nargs='*', default=None,
        help='Checkpoint iteration list forwarded to free-gaussians refinement training.'
    )
    parser.add_argument(
        '--dense_depth_output_mode',
        type=str,
        default='surf',
        choices=['expected', 'surf'],
        help='Depth export mode used by render_dense_views during the dense-view stage.',
    )
    args = parser.parse_args()
    
    # Set output paths
    if args.output_path is None:
        if args.source_path.endswith(os.sep):
            output_dir_name = args.source_path.split(os.sep)[-2]
        else:
            output_dir_name = args.source_path.split(os.sep)[-1]
        args.output_path = os.path.join('output', output_dir_name)
    mast3r_scene_path = os.path.join(args.output_path, 'mast3r_sfm')
    aligned_charts_path = os.path.join(args.output_path, 'mast3r_sfm')
    free_gaussians_path = os.path.join(args.output_path, 'free_gaussians')
    tsdf_meshes_path = os.path.join(args.output_path, 'tsdf_meshes')
    tetra_meshes_path = os.path.join(args.output_path, 'tetra_meshes')

    dense_view_idx_list = None
    if args.use_dense_view:
        _, dense_view_idx_list = load_or_create_dense_view_indices(args.source_path)
    
    # NOTE: Not use dense supervision from MAtCha
    dense_arg = ""
    
    # Free Gaussians refinement default config
    if args.free_gaussians_config is None:
        args.free_gaussians_config = 'long' if args.dense_supervision else 'default'

    n_images, image_idx_list, initial_selection_source = resolve_initial_view_selection(
        use_view_config=args.use_view_config,
        source_path=args.source_path,
        config_view_num=args.config_view_num,
        n_images=args.n_images,
        image_idx=args.image_idx,
        use_dense_view=args.use_dense_view,
        dense_view_idx_list=dense_view_idx_list,
    )
    if initial_selection_source != "all_images":
        print(f"[INFO] Initial run_sfm/align_charts selection source: {initial_selection_source}")
    
    # Defining commands
    sfm_command = " ".join([
        "python", "scripts/run_sfm.py",
        "--source_path", args.source_path,
        "--output_path", mast3r_scene_path,
        "--config", args.sfm_config,
        # "--env", args.sfm_env,
        "--n_images" if n_images is not None else "", str(n_images) if n_images is not None else "",
        "--image_idx" if image_idx_list is not None else "", " ".join([str(i) for i in image_idx_list]) if image_idx_list is not None else "",
        "--randomize_images" if args.randomize_images else "",
    ])
    
    align_charts_command_parts = [
        "python", "scripts/align_charts.py",
        "--source_path", mast3r_scene_path,
        "--mast3r_scene", mast3r_scene_path,
        "--output_path", aligned_charts_path,
        "--config", args.alignment_config,
        "--resolution", str(args.resolution),
        "--depth_model", args.depth_model,
        "--geometry_prior_mode", args.geometry_prior_mode,
        "--depthanythingv2_checkpoint_dir", args.depthanythingv2_checkpoint_dir,
        "--depthanything_encoder", args.depthanything_encoder,
        "--geometrycrafter_repo", args.geometrycrafter_repo,
        "--geometrycrafter_num_views", str(args.geometrycrafter_num_views),
        "--geometrycrafter_view_order", args.geometrycrafter_view_order,
        "--geometrycrafter_model_type", args.geometrycrafter_model_type,
        "--geometrycrafter_downsample_ratio", str(args.geometrycrafter_downsample_ratio),
        "--geometrycrafter_num_inference_steps", str(args.geometrycrafter_num_inference_steps),
        "--geometrycrafter_guidance_scale", str(args.geometrycrafter_guidance_scale),
        "--geometrycrafter_window_size", str(args.geometrycrafter_window_size),
        "--geometrycrafter_decode_chunk_size", str(args.geometrycrafter_decode_chunk_size),
        "--geometrycrafter_overlap", str(args.geometrycrafter_overlap),
        "--geometrycrafter_process_length", str(args.geometrycrafter_process_length),
        "--geometrycrafter_process_stride", str(args.geometrycrafter_process_stride),
        "--geometrycrafter_seed", str(args.geometrycrafter_seed),
        "--geometrycrafter_force_projection" if args.geometrycrafter_force_projection else "--geometrycrafter_no_force_projection",
        "--geometrycrafter_force_fixed_focal" if args.geometrycrafter_force_fixed_focal else "--geometrycrafter_no_force_fixed_focal",
    ]
    append_optional_arg(align_charts_command_parts, "--geometrycrafter_cache_root", args.geometrycrafter_cache_root)
    append_optional_arg(align_charts_command_parts, "--geometrycrafter_height", args.geometrycrafter_height)
    append_optional_arg(align_charts_command_parts, "--geometrycrafter_width", args.geometrycrafter_width)
    append_optional_arg(align_charts_command_parts, "--geometrycrafter_use_extract_interp", args.geometrycrafter_use_extract_interp)
    append_optional_arg(align_charts_command_parts, "--geometrycrafter_track_time", args.geometrycrafter_track_time)
    append_optional_arg(align_charts_command_parts, "--geometrycrafter_low_memory_usage", args.geometrycrafter_low_memory_usage)
    append_optional_arg(align_charts_command_parts, "--geometrycrafter_parallel_sequences", args.geometrycrafter_parallel_sequences)
    append_optional_arg(align_charts_command_parts, "--n_charts", n_images)
    append_optional_arg(
        align_charts_command_parts,
        "--image_indices",
        serialize_image_indices(image_idx_list),
    )
    append_optional_arg(
        align_charts_command_parts,
        "--colmap_scene",
        args.source_path if args.sfm_config == "posed" else None,
    )
    align_charts_command = " ".join(align_charts_command_parts)
    
    # NOTE: hard code plane-refine-depths path
    plane_root_path = os.path.join(mast3r_scene_path, 'plane-refine-depths')

    def build_refine_free_gaussians_command(include_camera_source_path: bool):
        refine_free_gaussians_command_parts = [
            "python", "scripts/refine_free_gaussians.py",
            "--mast3r_scene", mast3r_scene_path,
            "--output_path", free_gaussians_path,
            "--config", args.free_gaussians_config,
            "--resolution", str(args.resolution),
            "--dense_regul", args.dense_regul,
            "--refine_depth_path", plane_root_path,
        ]
        append_optional_arg(
            refine_free_gaussians_command_parts,
            "--camera_source_path",
            args.source_path if include_camera_source_path and args.sfm_config == "posed" else None,
        )
        if dense_arg:
            refine_free_gaussians_command_parts.append(dense_arg)
        if args.use_downsample_gaussians:
            refine_free_gaussians_command_parts.append("--use_downsample_gaussians")
        append_optional_arg(refine_free_gaussians_command_parts, "--mip_filter_variance", args.mip_filter_variance)
        if args.checkpoint_iterations:
            refine_free_gaussians_command_parts.extend([
                "--checkpoint_iterations",
                *[str(iteration) for iteration in args.checkpoint_iterations],
            ])
        return " ".join(refine_free_gaussians_command_parts)

    refine_free_gaussians_command = build_refine_free_gaussians_command(include_camera_source_path=True)
    refine_free_gaussians_dense_view_command = build_refine_free_gaussians_command(include_camera_source_path=False)

    render_all_img_command = " ".join([
        "python", "2d-gaussian-splatting/render_multires.py",
        "--source_path", mast3r_scene_path,
        "--model_path", free_gaussians_path,
        "--resolution", str(args.resolution),
        "--skip_test",
        "--skip_mesh",
        "--render_all_img",
        "--use_default_output_dir",
    ])
    if args.export_workers is not None:
        render_all_img_command = " ".join([render_all_img_command, "--export_workers", str(args.export_workers)])
    
    tsdf_command = " ".join([
        "python", "scripts/extract_tsdf_mesh.py",
        "--mast3r_scene", mast3r_scene_path,
        "--model_path", free_gaussians_path,
        "--output_path", tsdf_meshes_path,
        "--config", args.tsdf_config,
    ])
    
    tetra_command = " ".join([
        "python", "scripts/extract_tetra_mesh.py",
        "--mast3r_scene", mast3r_scene_path,
        "--model_path", free_gaussians_path,
        "--output_path", tetra_meshes_path,
        "--config", args.tetra_config,
        "--downsample_ratio", str(args.tetra_downsample_ratio),
        "--interpolate_views" if not args.no_interpolated_views else "",
        dense_arg,
    ])

    def get_see3d_inpaint_command(stage, select_inpaint_num):
        return " ".join([
        "python", "scripts/see3d_inpaint.py",
        "--source_path", mast3r_scene_path,
        "--model_path", free_gaussians_path,
        "--plane_root_dir", plane_root_path,
        "--resolution", str(args.resolution),
        "--iteration", '7000',
        "--see3d_stage", str(stage),
        "--select_inpaint_num", str(select_inpaint_num),
    ])

    eval_command = " ".join([
        "python", "2d-gaussian-splatting/eval/eval.py",
        "--source_path", args.source_path,
        "--model_path", args.output_path,
        "--sparse_view_num", str(args.config_view_num),
    ])

    render_charts_command = " ".join([
        "python", "2d-gaussian-splatting/render_chart_views.py",
        "--source_path", mast3r_scene_path,
        "--save_root_path", plane_root_path,
        "--resolution", str(args.resolution),
    ])
    if args.sfm_config == "posed":
        render_charts_command = " ".join([
            render_charts_command,
            "--camera_source_path", args.source_path,
        ])

    generate_2Dplane_command = " ".join([
        "python", "2d-gaussian-splatting/planes/plane_excavator.py",
        "--plane_root_path", plane_root_path,
        "--num_views", "12",
    ])

    pnts_path = os.path.join(mast3r_scene_path, 'chart_pcd.ply')
    vis_plane_path = os.path.join(mast3r_scene_path, 'vis_plane')

    def get_plane_refine_depth_command(anchor_view_id_json_path=None, see3d_root_path=None, include_camera_source_path=True):
        if see3d_root_path is not None:
            if anchor_view_id_json_path is not None:
                command = " ".join([
                    "python", "scripts/plane_refine_depth.py",
                    "--source_path", mast3r_scene_path,
                    "--plane_root_path", plane_root_path,
                    "--pnts_path", pnts_path,
                    "--resolution", str(args.resolution),
                    "--merge_resolution_scale", str(args.merge_resolution_scale),
                    "--merge_device", args.merge_device,
                    "--anchor_view_id_json_path", anchor_view_id_json_path,
                    "--see3d_root_path", see3d_root_path,
                ])
            else:
                command = " ".join([
                    "python", "scripts/plane_refine_depth.py",
                    "--source_path", mast3r_scene_path,
                    "--plane_root_path", plane_root_path,
                    "--pnts_path", pnts_path,
                    "--resolution", str(args.resolution),
                    "--merge_resolution_scale", str(args.merge_resolution_scale),
                    "--merge_device", args.merge_device,
                    "--see3d_root_path", see3d_root_path,
                ])
        else:
            command = " ".join([
                "python", "scripts/plane_refine_depth.py",
                "--source_path", mast3r_scene_path,
                "--plane_root_path", plane_root_path,
                "--pnts_path", pnts_path,
                "--resolution", str(args.resolution),
                    "--merge_resolution_scale", str(args.merge_resolution_scale),
                    "--merge_device", args.merge_device,
                ])
        append_optional_arg(
            command_parts := command.split(),
            "--camera_source_path",
            args.source_path if include_camera_source_path and args.sfm_config == "posed" else None,
        )
        append_optional_arg(
            command_parts,
            "--artifact_source_path",
            mast3r_scene_path if include_camera_source_path and args.sfm_config == "posed" else None,
        )
        command = " ".join(command_parts)
        return command
        
    see3d_root_path = os.path.join(mast3r_scene_path, 'see3d_render')

    render_eval_path = os.path.join(free_gaussians_path, 'train', 'ours_7000', 'renders')

    t1 = time.time()
    skip_initial_chart_plane_refine = should_skip_initial_chart_plane_refine(
        args.depth_model,
        args.geometry_prior_mode,
    )
    
    # run MAtCha training
    run_command_safe(sfm_command)
    run_command_safe(align_charts_command)

    # generate 2D planes + refine depth for input views + init gaussian training
    run_command_safe(render_charts_command)
    if skip_initial_chart_plane_refine:
        print("[INFO] Skipping initial chart-stage plane refinement for GeometryCrafter bootstrap; using chart depths directly.")
    else:
        run_command_safe(generate_2Dplane_command)
        run_command_safe(get_plane_refine_depth_command(anchor_view_id_json_path=None, see3d_root_path=None))
    run_command_safe(refine_free_gaussians_command)

    if args.use_dense_view:
        # replace the sparse/0 with dense-view-sparse/0, use dense view for training
        # copy point3D files from sparse/0 to dense-view-sparse/0
        copy_sparse_point_files(
            f'{mast3r_scene_path}/sparse/0',
            f'{mast3r_scene_path}/dense-view-sparse/0',
        )

        # render dense views
        render_dense_views_command = " ".join([
            "python", "2d-gaussian-splatting/render_dense_views.py",
            "--source_path", mast3r_scene_path,
            "--model_path", free_gaussians_path,
            "--resolution", str(args.resolution),
            "--iteration", "7000",
            "--depth_output_mode", args.dense_depth_output_mode,
        ])
        run_command_safe(render_dense_views_command)

        # generate depth and normal for dense views
        gen_dn_dense_views_command = " ".join([
            "python", "2d-gaussian-splatting/guidance/dense_gc_util.py",
            "--source_path", mast3r_scene_path,
            "--model_path", free_gaussians_path,
            "--iteration", "7000",
            "--geometrycrafter_repo", args.geometrycrafter_repo,
            "--geometrycrafter_num_views", str(args.geometrycrafter_num_views),
            "--geometrycrafter_view_order", args.geometrycrafter_view_order,
            "--geometrycrafter_model_type", args.geometrycrafter_model_type,
            "--geometrycrafter_height", str(args.geometrycrafter_height),
            "--geometrycrafter_width", str(args.geometrycrafter_width),
            "--geometrycrafter_downsample_ratio", str(args.geometrycrafter_downsample_ratio),
            "--geometrycrafter_num_inference_steps", str(args.geometrycrafter_num_inference_steps),
            "--geometrycrafter_guidance_scale", str(args.geometrycrafter_guidance_scale),
            "--geometrycrafter_window_size", str(args.geometrycrafter_window_size),
            "--geometrycrafter_decode_chunk_size", str(args.geometrycrafter_decode_chunk_size),
            "--geometrycrafter_overlap", str(args.geometrycrafter_overlap),
            "--geometrycrafter_process_length", str(args.geometrycrafter_process_length),
            "--geometrycrafter_process_stride", str(args.geometrycrafter_process_stride),
            "--geometrycrafter_seed", str(args.geometrycrafter_seed),
            "--geometrycrafter_parallel_sequences", str(args.geometrycrafter_parallel_sequences),
            "--geometrycrafter_force_projection" if args.geometrycrafter_force_projection else "--geometrycrafter_no_force_projection",
            "--geometrycrafter_force_fixed_focal" if args.geometrycrafter_force_fixed_focal else "--geometrycrafter_no_force_fixed_focal",
        ])
        if args.geometrycrafter_cache_root is not None:
            gen_dn_dense_views_command = " ".join([gen_dn_dense_views_command, "--geometrycrafter_cache_root", args.geometrycrafter_cache_root])
        if args.geometrycrafter_use_extract_interp:
            gen_dn_dense_views_command = " ".join([gen_dn_dense_views_command, "--geometrycrafter_use_extract_interp"])
        if args.geometrycrafter_track_time:
            gen_dn_dense_views_command = " ".join([gen_dn_dense_views_command, "--geometrycrafter_track_time"])
        if args.geometrycrafter_low_memory_usage:
            gen_dn_dense_views_command = " ".join([gen_dn_dense_views_command, "--geometrycrafter_low_memory_usage"])
        run_command_safe(gen_dn_dense_views_command)

        run_command_safe(generate_2Dplane_command)
        run_command_safe(get_plane_refine_depth_command(anchor_view_id_json_path=None, see3d_root_path=None, include_camera_source_path=False))
        mv_cmd = f'mv {free_gaussians_path}/point_cloud {free_gaussians_path}/point_cloud-chart-views'
        run_command_safe(mv_cmd)
        run_command_safe(refine_free_gaussians_dense_view_command)

        # render all images, export mesh, and evaluate
        if not args.skip_render_all_img:
            run_command_safe(render_all_img_command)
        run_command_safe(tetra_command)

        print("Finished training dense view without See3D prior!")

        t2 = time.time()
        print(f"Total running time: {t2 - t1} seconds")
        exit()


    # see3d inpainting stage 1 + refine depth with 2D planes + continue gaussian training
    run_command_safe(get_see3d_inpaint_command(1, args.select_inpaint_num))
    run_command_safe(get_plane_refine_depth_command(anchor_view_id_json_path=None, see3d_root_path=see3d_root_path))
    mv_cmd = f'mv {free_gaussians_path}/point_cloud {free_gaussians_path}/point_cloud-ori'
    run_command_safe(mv_cmd)
    run_command_safe(refine_free_gaussians_command)

    # see3d inpainting stage 2 + refine depth with 2D planes + continue gaussian training
    run_command_safe(get_see3d_inpaint_command(2, args.select_inpaint_num))
    run_command_safe(get_plane_refine_depth_command(anchor_view_id_json_path=None, see3d_root_path=see3d_root_path))
    mv_cmd = f'mv {free_gaussians_path}/point_cloud {free_gaussians_path}/point_cloud-s1'
    run_command_safe(mv_cmd)
    run_command_safe(refine_free_gaussians_command)

    # see3d inpainting stage 3 + refine depth with 2D planes + continue gaussian training
    run_command_safe(get_see3d_inpaint_command(3, args.select_inpaint_num))
    anchor_view_id_json_path = os.path.join(see3d_root_path, 'stage3', 'anchor_view_id.json')
    run_command_safe(get_plane_refine_depth_command(anchor_view_id_json_path=anchor_view_id_json_path, see3d_root_path=see3d_root_path))
    mv_cmd = f'mv {free_gaussians_path}/point_cloud {free_gaussians_path}/point_cloud-s2'
    run_command_safe(mv_cmd)
    run_command_safe(refine_free_gaussians_command)

    # render all images, export mesh, and evaluate
    if not args.skip_render_all_img:
        run_command_safe(render_all_img_command)
    run_command_safe(tetra_command)

    if args.use_mesh_filter:
        # use mesh filter for forward facing scene
        mesh_path = os.path.join(tetra_meshes_path, 'tetra_mesh_binary_search_7_iter_7000.ply')
        length_threshold = 0.5
        filtered_mesh_path = os.path.join(tetra_meshes_path, f'tetra_mesh_binary_search_7_iter_7000_filtered_t{length_threshold}.ply')
        filter_mesh_command = " ".join([
            "python", "2d-gaussian-splatting/utils/mesh_filter.py",
            "--mesh_path", mesh_path,
            "--output_path", filtered_mesh_path,
        ])
        run_command_safe(filter_mesh_command)
        mv_cmd = f'mv {mesh_path} {tetra_meshes_path}/tetra_mesh_binary_search_7_iter_7000_ori.ply'
        run_command_safe(mv_cmd)
        mv_cmd = f'mv {filtered_mesh_path} {mesh_path}'
        run_command_safe(mv_cmd)

    run_command_safe(eval_command)

    # # vis global 3D plane by mesh (NOTE: slightly slow)
    # mesh_list = os.listdir(tetra_meshes_path)
    # mesh_list = [mesh_name for mesh_name in mesh_list if mesh_name.endswith('.ply')]
    # mesh_list.sort()
    # mesh_name = mesh_list[-1]
    # mesh_path = os.path.join(tetra_meshes_path, mesh_name)
    # print(f"Mesh path: {mesh_path}")
    # vis_global_3Dplane_by_mesh_command = " ".join([
    #     "python", "2d-gaussian-splatting/planes/vis_global_3Dplane_by_mesh.py",
    #     "--source_path", mast3r_scene_path,
    #     "--mesh_path", mesh_path,
    #     "--plane_root_path", plane_root_path,
    #     "--see3d_root_path", see3d_root_path,
    #     "--output_path", os.path.join(args.output_path, 'vis_global_plane_color_mesh.ply'),
    # ])
    # run_command_safe(vis_global_3Dplane_by_mesh_command)

    t2 = time.time()
    print(f"Total running time: {t2 - t1} seconds")
