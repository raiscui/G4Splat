#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/rerun_dense_view_stage.sh <output_root> [options]

Options:
  --config <name>                  Free-gaussians config preset. Default: dense_compact
  --iteration <n>                  Iteration to restore and render from. Default: 7000
  --backup-dir <path>              Explicit backup iteration directory to restore.
  --dense-regul <name>             Dense regularization strength. Default: default
  --checkpoint-iterations <ints>   Forward checkpoint iteration list to refine_free_gaussians.py
  --mip-filter-variance <float>    Override mip filter strength during dense refinement
  --tetra-config <name>            Tetra extraction config. Default: default
  --tetra-downsample-ratio <f>     Tetra downsample ratio. Default: 0.25
  --skip-render-all-img            Skip render_multires export before tetra extraction
  --export-workers <n>             Parallel workers for render_multires image export
  --geometrycrafter-repo <path>    GeometryCrafter repo path. Default: /home/rais/GeometryCrafter
  --geometrycrafter-cache-root <path>
  --geometrycrafter-num-views <n>  Interleaved view count. Default: 12
  --geometrycrafter-view-order <csv>
  --geometrycrafter-model-type <diff|determ>
  --geometrycrafter-height <n>     Processing height. Default: 576
  --geometrycrafter-width <n>      Processing width. Default: 1024
  --geometrycrafter-downsample-ratio <f>
  --geometrycrafter-num-inference-steps <n>
  --geometrycrafter-guidance-scale <f>
  --geometrycrafter-window-size <n>
  --geometrycrafter-decode-chunk-size <n>
  --geometrycrafter-overlap <n>
  --geometrycrafter-process-length <n>
  --geometrycrafter-process-stride <n>
  --geometrycrafter-seed <n>
  --geometrycrafter-parallel-sequences <n>
  --geometrycrafter-no-force-projection
  --geometrycrafter-no-force-fixed-focal
  --geometrycrafter-use-extract-interp
  --geometrycrafter-track-time
  --geometrycrafter-low-memory-usage
  --no-interpolate-views           Disable tetra interpolate views flag.

The script reuses the existing MASt3R/charts outputs under <output_root>/mast3r_sfm
and only reruns the dense-view refinement stage.
EOF
}

run_cmd() {
  echo
  echo "[RUN] $*"
  "$@"
}

require_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    echo "[ERROR] Missing directory: $path" >&2
    exit 1
  fi
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "[ERROR] Missing file: $path" >&2
    exit 1
  fi
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

OUTPUT_ROOT="$1"
shift

CONFIG="dense_compact"
ITERATION="7000"
BACKUP_DIR=""
DENSE_REGUL="default"
CHECKPOINT_ITERATIONS=()
MIP_FILTER_VARIANCE=""
TETRA_CONFIG="default"
TETRA_DOWNSAMPLE_RATIO="0.25"
INTERPOLATE_VIEWS=1
SKIP_RENDER_ALL_IMG=0
EXPORT_WORKERS=""
GEOMETRYCRAFTER_REPO="/home/rais/GeometryCrafter"
GEOMETRYCRAFTER_CACHE_ROOT=""
GEOMETRYCRAFTER_NUM_VIEWS="12"
GEOMETRYCRAFTER_VIEW_ORDER="0,1,10,11,2,3,4,5,6,7,8,9"
GEOMETRYCRAFTER_MODEL_TYPE="diff"
GEOMETRYCRAFTER_HEIGHT="576"
GEOMETRYCRAFTER_WIDTH="1024"
GEOMETRYCRAFTER_DOWNSAMPLE_RATIO="1.0"
GEOMETRYCRAFTER_NUM_INFERENCE_STEPS="5"
GEOMETRYCRAFTER_GUIDANCE_SCALE="1.0"
GEOMETRYCRAFTER_WINDOW_SIZE="110"
GEOMETRYCRAFTER_DECODE_CHUNK_SIZE="8"
GEOMETRYCRAFTER_OVERLAP="25"
GEOMETRYCRAFTER_PROCESS_LENGTH="-1"
GEOMETRYCRAFTER_PROCESS_STRIDE="1"
GEOMETRYCRAFTER_SEED="42"
GEOMETRYCRAFTER_PARALLEL_SEQUENCES="1"
GEOMETRYCRAFTER_FORCE_PROJECTION=1
GEOMETRYCRAFTER_FORCE_FIXED_FOCAL=1
GEOMETRYCRAFTER_USE_EXTRACT_INTERP=0
GEOMETRYCRAFTER_TRACK_TIME=0
GEOMETRYCRAFTER_LOW_MEMORY_USAGE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --iteration)
      ITERATION="$2"
      shift 2
      ;;
    --backup-dir)
      BACKUP_DIR="$2"
      shift 2
      ;;
    --dense-regul)
      DENSE_REGUL="$2"
      shift 2
      ;;
    --checkpoint-iterations)
      shift
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        CHECKPOINT_ITERATIONS+=("$1")
        shift
      done
      ;;
    --mip-filter-variance)
      MIP_FILTER_VARIANCE="$2"
      shift 2
      ;;
    --tetra-config)
      TETRA_CONFIG="$2"
      shift 2
      ;;
    --tetra-downsample-ratio)
      TETRA_DOWNSAMPLE_RATIO="$2"
      shift 2
      ;;
    --skip-render-all-img)
      SKIP_RENDER_ALL_IMG=1
      shift
      ;;
    --export-workers)
      EXPORT_WORKERS="$2"
      shift 2
      ;;
    --geometrycrafter-repo)
      GEOMETRYCRAFTER_REPO="$2"
      shift 2
      ;;
    --geometrycrafter-cache-root)
      GEOMETRYCRAFTER_CACHE_ROOT="$2"
      shift 2
      ;;
    --geometrycrafter-num-views)
      GEOMETRYCRAFTER_NUM_VIEWS="$2"
      shift 2
      ;;
    --geometrycrafter-view-order)
      GEOMETRYCRAFTER_VIEW_ORDER="$2"
      shift 2
      ;;
    --geometrycrafter-model-type)
      GEOMETRYCRAFTER_MODEL_TYPE="$2"
      shift 2
      ;;
    --geometrycrafter-height)
      GEOMETRYCRAFTER_HEIGHT="$2"
      shift 2
      ;;
    --geometrycrafter-width)
      GEOMETRYCRAFTER_WIDTH="$2"
      shift 2
      ;;
    --geometrycrafter-downsample-ratio)
      GEOMETRYCRAFTER_DOWNSAMPLE_RATIO="$2"
      shift 2
      ;;
    --geometrycrafter-num-inference-steps)
      GEOMETRYCRAFTER_NUM_INFERENCE_STEPS="$2"
      shift 2
      ;;
    --geometrycrafter-guidance-scale)
      GEOMETRYCRAFTER_GUIDANCE_SCALE="$2"
      shift 2
      ;;
    --geometrycrafter-window-size)
      GEOMETRYCRAFTER_WINDOW_SIZE="$2"
      shift 2
      ;;
    --geometrycrafter-decode-chunk-size)
      GEOMETRYCRAFTER_DECODE_CHUNK_SIZE="$2"
      shift 2
      ;;
    --geometrycrafter-overlap)
      GEOMETRYCRAFTER_OVERLAP="$2"
      shift 2
      ;;
    --geometrycrafter-process-length)
      GEOMETRYCRAFTER_PROCESS_LENGTH="$2"
      shift 2
      ;;
    --geometrycrafter-process-stride)
      GEOMETRYCRAFTER_PROCESS_STRIDE="$2"
      shift 2
      ;;
    --geometrycrafter-seed)
      GEOMETRYCRAFTER_SEED="$2"
      shift 2
      ;;
    --geometrycrafter-parallel-sequences)
      GEOMETRYCRAFTER_PARALLEL_SEQUENCES="$2"
      shift 2
      ;;
    --geometrycrafter-no-force-projection)
      GEOMETRYCRAFTER_FORCE_PROJECTION=0
      shift
      ;;
    --geometrycrafter-no-force-fixed-focal)
      GEOMETRYCRAFTER_FORCE_FIXED_FOCAL=0
      shift
      ;;
    --geometrycrafter-use-extract-interp)
      GEOMETRYCRAFTER_USE_EXTRACT_INTERP=1
      shift
      ;;
    --geometrycrafter-track-time)
      GEOMETRYCRAFTER_TRACK_TIME=1
      shift
      ;;
    --geometrycrafter-low-memory-usage)
      GEOMETRYCRAFTER_LOW_MEMORY_USAGE=1
      shift
      ;;
    --no-interpolate-views)
      INTERPOLATE_VIEWS=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

OUTPUT_ROOT="${OUTPUT_ROOT%/}"
MASTR3_SCENE="${OUTPUT_ROOT}/mast3r_sfm"
FREE_GAUSSIANS_DIR="${OUTPUT_ROOT}/free_gaussians"
POINT_CLOUD_DIR="${FREE_GAUSSIANS_DIR}/point_cloud"
TRAIN_DIR="${FREE_GAUSSIANS_DIR}/train/ours_${ITERATION}"
TETRA_MESHES_DIR="${OUTPUT_ROOT}/tetra_meshes"

require_dir "$MASTR3_SCENE"
require_dir "$FREE_GAUSSIANS_DIR"
require_dir "${MASTR3_SCENE}/sparse/0"
require_file "${MASTR3_SCENE}/chart_pcd.ply"
require_file "configs/free_gaussians_refinement/${CONFIG}.yaml"

if [[ -z "$BACKUP_DIR" ]]; then
  if [[ -d "${FREE_GAUSSIANS_DIR}/point_cloud-chart-views/iteration_${ITERATION}" ]]; then
    BACKUP_DIR="${FREE_GAUSSIANS_DIR}/point_cloud-chart-views/iteration_${ITERATION}"
  elif [[ -d "${FREE_GAUSSIANS_DIR}/point_cloud-chart-views/point_cloud/iteration_${ITERATION}" ]]; then
    BACKUP_DIR="${FREE_GAUSSIANS_DIR}/point_cloud-chart-views/point_cloud/iteration_${ITERATION}"
  else
    echo "[ERROR] Could not find a dense-view backup iteration directory." >&2
    echo "        Pass --backup-dir explicitly." >&2
    exit 1
  fi
fi

require_dir "$BACKUP_DIR"
require_file "${BACKUP_DIR}/point_cloud.ply"

mkdir -p "${MASTR3_SCENE}/dense-view-sparse/0"
for ext in bin txt ply; do
  run_cmd cp -f "${MASTR3_SCENE}/sparse/0/points3D.${ext}" "${MASTR3_SCENE}/dense-view-sparse/0/points3D.${ext}"
done

echo
echo "[INFO] Output root: $OUTPUT_ROOT"
echo "[INFO] Restore backup: $BACKUP_DIR"
echo "[INFO] Free-gaussians config: $CONFIG"
echo "[INFO] Iteration: $ITERATION"
echo "[INFO] Dense regul: $DENSE_REGUL"
echo "[INFO] GeometryCrafter parallel sequences: $GEOMETRYCRAFTER_PARALLEL_SEQUENCES"

run_cmd rm -rf "$POINT_CLOUD_DIR"
run_cmd rm -rf "$TRAIN_DIR"
run_cmd rm -rf "${MASTR3_SCENE}/render-dense-train-views"
run_cmd rm -rf "${MASTR3_SCENE}/plane-refine-depths"
run_cmd rm -rf "$TETRA_MESHES_DIR"

run_cmd mkdir -p "$POINT_CLOUD_DIR"
run_cmd cp -a "$BACKUP_DIR" "$POINT_CLOUD_DIR/"

run_cmd pixi run python 2d-gaussian-splatting/render_dense_views.py \
  --source_path "$MASTR3_SCENE" \
  --model_path "$FREE_GAUSSIANS_DIR" \
  --iteration "$ITERATION"

GC_CMD=(
  pixi run python 2d-gaussian-splatting/guidance/dense_gc_util.py
  --source_path "$MASTR3_SCENE"
  --model_path "$FREE_GAUSSIANS_DIR"
  --iteration "$ITERATION"
  --geometrycrafter_repo "$GEOMETRYCRAFTER_REPO"
  --geometrycrafter_num_views "$GEOMETRYCRAFTER_NUM_VIEWS"
  --geometrycrafter_view_order "$GEOMETRYCRAFTER_VIEW_ORDER"
  --geometrycrafter_model_type "$GEOMETRYCRAFTER_MODEL_TYPE"
  --geometrycrafter_height "$GEOMETRYCRAFTER_HEIGHT"
  --geometrycrafter_width "$GEOMETRYCRAFTER_WIDTH"
  --geometrycrafter_downsample_ratio "$GEOMETRYCRAFTER_DOWNSAMPLE_RATIO"
  --geometrycrafter_num_inference_steps "$GEOMETRYCRAFTER_NUM_INFERENCE_STEPS"
  --geometrycrafter_guidance_scale "$GEOMETRYCRAFTER_GUIDANCE_SCALE"
  --geometrycrafter_window_size "$GEOMETRYCRAFTER_WINDOW_SIZE"
  --geometrycrafter_decode_chunk_size "$GEOMETRYCRAFTER_DECODE_CHUNK_SIZE"
  --geometrycrafter_overlap "$GEOMETRYCRAFTER_OVERLAP"
  --geometrycrafter_process_length "$GEOMETRYCRAFTER_PROCESS_LENGTH"
  --geometrycrafter_process_stride "$GEOMETRYCRAFTER_PROCESS_STRIDE"
  --geometrycrafter_seed "$GEOMETRYCRAFTER_SEED"
  --geometrycrafter_parallel_sequences "$GEOMETRYCRAFTER_PARALLEL_SEQUENCES"
)
if [[ -n "$GEOMETRYCRAFTER_CACHE_ROOT" ]]; then
  GC_CMD+=(--geometrycrafter_cache_root "$GEOMETRYCRAFTER_CACHE_ROOT")
fi
if [[ "$GEOMETRYCRAFTER_FORCE_PROJECTION" -eq 1 ]]; then
  GC_CMD+=(--geometrycrafter_force_projection)
else
  GC_CMD+=(--geometrycrafter_no_force_projection)
fi
if [[ "$GEOMETRYCRAFTER_FORCE_FIXED_FOCAL" -eq 1 ]]; then
  GC_CMD+=(--geometrycrafter_force_fixed_focal)
else
  GC_CMD+=(--geometrycrafter_no_force_fixed_focal)
fi
if [[ "$GEOMETRYCRAFTER_USE_EXTRACT_INTERP" -eq 1 ]]; then
  GC_CMD+=(--geometrycrafter_use_extract_interp)
fi
if [[ "$GEOMETRYCRAFTER_TRACK_TIME" -eq 1 ]]; then
  GC_CMD+=(--geometrycrafter_track_time)
fi
if [[ "$GEOMETRYCRAFTER_LOW_MEMORY_USAGE" -eq 1 ]]; then
  GC_CMD+=(--geometrycrafter_low_memory_usage)
fi
run_cmd "${GC_CMD[@]}"

run_cmd pixi run python 2d-gaussian-splatting/planes/plane_excavator.py \
  --plane_root_path "${MASTR3_SCENE}/plane-refine-depths" \
  --num_views 12

run_cmd pixi run python scripts/plane_refine_depth.py \
  --source_path "$MASTR3_SCENE" \
  --plane_root_path "${MASTR3_SCENE}/plane-refine-depths" \
  --pnts_path "${MASTR3_SCENE}/chart_pcd.ply"

REFINE_CMD=(
  pixi run python scripts/refine_free_gaussians.py
  --mast3r_scene "$MASTR3_SCENE"
  --output_path "$FREE_GAUSSIANS_DIR"
  --config "$CONFIG"
  --dense_regul "$DENSE_REGUL"
  --refine_depth_path "${MASTR3_SCENE}/plane-refine-depths"
)
if [[ -n "$MIP_FILTER_VARIANCE" ]]; then
  REFINE_CMD+=(--mip_filter_variance "$MIP_FILTER_VARIANCE")
fi
if [[ ${#CHECKPOINT_ITERATIONS[@]} -gt 0 ]]; then
  REFINE_CMD+=(--checkpoint_iterations "${CHECKPOINT_ITERATIONS[@]}")
fi
run_cmd "${REFINE_CMD[@]}"

if [[ "$SKIP_RENDER_ALL_IMG" -eq 0 ]]; then
  RENDER_CMD=(
    pixi run python 2d-gaussian-splatting/render_multires.py
    --source_path "$MASTR3_SCENE"
    --model_path "$FREE_GAUSSIANS_DIR"
    --skip_test
    --skip_mesh
    --render_all_img
    --use_default_output_dir
  )
  if [[ -n "$EXPORT_WORKERS" ]]; then
    RENDER_CMD+=(--export_workers "$EXPORT_WORKERS")
  fi
  run_cmd "${RENDER_CMD[@]}"
fi

TETRA_CMD=(
  pixi run python scripts/extract_tetra_mesh.py
  --mast3r_scene "$MASTR3_SCENE"
  --model_path "$FREE_GAUSSIANS_DIR"
  --output_path "$TETRA_MESHES_DIR"
  --config "$TETRA_CONFIG"
  --downsample_ratio "$TETRA_DOWNSAMPLE_RATIO"
)

if [[ "$INTERPOLATE_VIEWS" -eq 1 ]]; then
  TETRA_CMD+=(--interpolate_views)
fi

run_cmd "${TETRA_CMD[@]}"

echo
echo "[DONE] Dense-view stage rerun complete."
