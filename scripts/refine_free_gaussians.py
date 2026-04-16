import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml

from rich.console import Console

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


def serialize_depth_order_schedule(schedule):
    if schedule is None:
        return None
    if isinstance(schedule, str):
        normalized = schedule.strip()
        return normalized or None

    serialized_items = []
    for item in schedule:
        if isinstance(item, dict):
            if "iteration" not in item or "weight" not in item:
                raise ValueError(
                    "depth_order_schedule dict items must contain 'iteration' and 'weight' keys."
                )
            iteration = int(item["iteration"])
            weight = float(item["weight"])
        else:
            if len(item) != 2:
                raise ValueError(
                    "depth_order_schedule items must be pairs like [iteration, weight]."
                )
            iteration = int(item[0])
            weight = float(item[1])
        serialized_items.append(f"{iteration}:{weight}")

    return ",".join(serialized_items) or None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Scene arguments
    parser.add_argument('-s', '--mast3r_scene', type=str, required=True)
    parser.add_argument('-o', '--output_path', type=str, default=None)
    parser.add_argument('--camera_source_path', type=str, default=None)
    parser.add_argument('--white_background', type=bool, default=False)
    
    # For dense RGB and depth supervision from a COLMAP dataset (optional)
    parser.add_argument("--dense_data_path", type=str, default=None)
    parser.add_argument('--depthanythingv2_checkpoint_dir', type=str, default='./Depth-Anything-V2/checkpoints/')
    parser.add_argument('--depthanything_encoder', type=str, default='vitl')
    parser.add_argument('--dense_regul', type=str, default='default', help='Strength of dense regularization. Can be "default", "strong", "weak", or "none".')
    
    # Config
    parser.add_argument('-c', '--config', type=str, default='default')
    parser.add_argument('--resolution', type=int, default=1,
                        help='Image downsampling factor forwarded to 2DGS training. Use 2 for half-resolution.')
    parser.add_argument('--checkpoint_iterations', type=int, nargs='*', default=None,
                        help='Checkpoint iteration list forwarded to 2DGS training.')
    parser.add_argument('--mip_filter_variance', type=float, default=None,
                        help='Override mip filter strength. Lower values preserve more distant detail.')
    parser.add_argument(
        '--depth_order_schedule',
        type=str,
        default=None,
        help="Override the depth order schedule with 'iteration:weight,...'.",
    )

    parser.add_argument('--refine_depth_path', type=str, default=None, help='Path to the refine depth directory')
    parser.add_argument('--use_downsample_gaussians', action='store_true', help='Use downsample gaussians')
    
    args = parser.parse_args()
    
    # Set console
    CONSOLE = Console(width=120)
    
    # Set output path
    if args.output_path is None:
        if args.source_path.endswith(os.sep):
            output_dir_name = args.source_path.split(os.sep)[-2]
        else:
            output_dir_name = args.source_path.split(os.sep)[-1]
        args.output_path = os.path.join('output', output_dir_name)
        args.output_path = os.path.join(args.output_path, 'refined_free_gaussians')
    os.makedirs(args.output_path, exist_ok=True)
    CONSOLE.print(f"[INFO] Refined free gaussians will be saved to: {args.output_path}")
    
    # Load config
    config_path = os.path.join('configs/free_gaussians_refinement', args.config + '.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Dense supervision (optional)
    if args.dense_data_path is not None:
        dense_arg = " ".join([
            "--dense_data_path", args.dense_data_path,
        ])
    else:
        dense_arg = ""

    # Define command
    if args.refine_depth_path is not None:
        print(f'refine depth path {args.refine_depth_path}, train gs use refine depth')
        command_parts = [
            "python", "2d-gaussian-splatting/train_with_refine_depth.py",
            "-s", args.mast3r_scene,
            "-m", args.output_path,
            "--iterations", str(config['iterations']),
            "--densify_until_iter", str(config['densify_until_iter']),
            "--opacity_reset_interval", str(config['opacity_reset_interval']),
            "--depth_ratio", str(config['depth_ratio']),
            "--resolution", str(args.resolution),
            "--normal_consistency_from", str(config['normal_consistency_from']),
            "--distortion_from", str(config['distortion_from']),
            "--depthanythingv2_checkpoint_dir", args.depthanythingv2_checkpoint_dir,
            "--depthanything_encoder", args.depthanything_encoder,
            "--dense_regul", args.dense_regul,
            "--refine_depth_path", args.refine_depth_path,
        ]
        append_optional_arg(command_parts, "--camera_source_path", args.camera_source_path)
        if dense_arg:
            command_parts.extend(dense_arg.split())

        append_optional_arg(command_parts, "--use_mip_filter", config.get("use_mip_filter", False))
        mip_filter_variance = args.mip_filter_variance
        if mip_filter_variance is None:
            mip_filter_variance = config.get("mip_filter_variance")
        append_optional_arg(command_parts, "--mip_filter_variance", mip_filter_variance)
        depth_order_schedule = args.depth_order_schedule
        if depth_order_schedule is None:
            depth_order_schedule = serialize_depth_order_schedule(config.get("depth_order_schedule"))
        append_optional_arg(command_parts, "--depth_order_schedule", depth_order_schedule)
        append_optional_arg(command_parts, "--densify_from_iter", config.get("densify_from_iter"))
        append_optional_arg(command_parts, "--densification_interval", config.get("densification_interval"))
        append_optional_arg(command_parts, "--densify_grad_threshold", config.get("densify_grad_threshold"))
        append_optional_arg(command_parts, "--opacity_cull", config.get("opacity_cull"))
        append_optional_arg(command_parts, "--percent_dense", config.get("percent_dense"))
        append_optional_arg(command_parts, "--max_init_gaussians", config.get("max_init_gaussians"))
        append_optional_arg(command_parts, "--init_voxel_size", config.get("init_voxel_size"))
        append_optional_arg(command_parts, "--max_init_input_views", config.get("max_init_input_views"))
        append_optional_arg(command_parts, "--init_point_stride", config.get("init_point_stride"))

        if args.checkpoint_iterations:
            command_parts.append("--checkpoint_iterations")
            command_parts.extend([str(iteration) for iteration in args.checkpoint_iterations])

        if args.use_downsample_gaussians or config.get("use_downsample_gaussians", False):
            command_parts.append("--use_downsample_gaussians")

        command = " ".join(command_parts)
    else:
        raise ValueError('refine depth path is required')
    
    # Run command
    CONSOLE.print(f"[INFO] Running command:\n{command}")
    run_command_safe(command)
