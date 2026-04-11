#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/continue_dense_view_stage.sh <mast3r_scene> <free_gaussians_dir> [options]

Arguments:
  <mast3r_scene>         Path to the MASt3R scene directory, e.g. /autodl-fs/data/g4/dm7_sr_v2/mast3r_sfm
  <free_gaussians_dir>   Path to the free-gaussians output directory, e.g. /autodl-fs/data/g4/dm7_sr_v2/free_gaussians_oomsafe

Options:
  --config <name>                  Free-gaussians config preset. Default: dense_compact_sm
  --iteration <n>                  Iteration to use for dense-view rendering/refinement. Default: 7000
  --resolution <n>                 Image-stage downsampling factor. Default: 1
  --tetra-config <name>            Tetra extraction config. Default: default
  --tetra-downsample-ratio <f>     Tetra downsample ratio. Default: 0.25
  --merge-resolution-scale <f>     plane_refine_depth merge resolution scale. Default: 2
  --merge-device <cpu|cuda>        plane_refine_depth merge device. Default: cuda
  --no-interpolate-views           Disable tetra interpolate views flag

This script assumes the first free-gaussians refinement has already succeeded and
that <free_gaussians_dir>/point_cloud/iteration_<iteration>/point_cloud.ply exists.
It then runs the dense-view continuation steps that train.py would normally execute
under --use_dense_view.
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

if [[ $# -ge 1 && ( "$1" == "-h" || "$1" == "--help" ) ]]; then
  usage
  exit 0
fi

if [[ $# -lt 2 ]]; then
  usage
  exit 1
fi

MASTR3_SCENE="${1%/}"
FREE_GAUSSIANS_DIR="${2%/}"
shift 2

CONFIG="dense_compact_sm"
ITERATION="7000"
RESOLUTION="1"
TETRA_CONFIG="default"
TETRA_DOWNSAMPLE_RATIO="0.25"
MERGE_RESOLUTION_SCALE="2"
MERGE_DEVICE="cuda"
INTERPOLATE_VIEWS=1

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
    --resolution)
      RESOLUTION="$2"
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
    --merge-resolution-scale)
      MERGE_RESOLUTION_SCALE="$2"
      shift 2
      ;;
    --merge-device)
      MERGE_DEVICE="$2"
      shift 2
      ;;
    --no-interpolate-views)
      INTERPOLATE_VIEWS=0
      shift
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

OUTPUT_ROOT="$(dirname "$MASTR3_SCENE")"
POINT_CLOUD_DIR="${FREE_GAUSSIANS_DIR}/point_cloud"
POINT_CLOUD_ITER_DIR="${POINT_CLOUD_DIR}/iteration_${ITERATION}"
POINT_CLOUD_CHART_DIR="${FREE_GAUSSIANS_DIR}/point_cloud-chart-views"
POINT_CLOUD_CHART_ITER_DIR="${POINT_CLOUD_CHART_DIR}/iteration_${ITERATION}"
TRAIN_DIR="${FREE_GAUSSIANS_DIR}/train/ours_${ITERATION}"
PLANE_ROOT_PATH="${MASTR3_SCENE}/plane-refine-depths"
TETRA_MESHES_DIR="${OUTPUT_ROOT}/tetra_meshes"

require_dir "$MASTR3_SCENE"
require_dir "$FREE_GAUSSIANS_DIR"
require_dir "${MASTR3_SCENE}/sparse/0"
require_file "${MASTR3_SCENE}/chart_pcd.ply"
require_file "configs/free_gaussians_refinement/${CONFIG}.yaml"

if [[ -f "${POINT_CLOUD_ITER_DIR}/point_cloud.ply" ]]; then
  SOURCE_POINT_CLOUD_DIR="$POINT_CLOUD_DIR"
elif [[ -f "${POINT_CLOUD_CHART_ITER_DIR}/point_cloud.ply" ]]; then
  SOURCE_POINT_CLOUD_DIR="$POINT_CLOUD_CHART_DIR"
else
  echo "[ERROR] Missing first-stage point cloud in both:" >&2
  echo "        ${POINT_CLOUD_ITER_DIR}/point_cloud.ply" >&2
  echo "        ${POINT_CLOUD_CHART_ITER_DIR}/point_cloud.ply" >&2
  exit 1
fi

mkdir -p "${MASTR3_SCENE}/dense-view-sparse/0"
for ext in bin txt ply; do
  run_cmd cp -f "${MASTR3_SCENE}/sparse/0/points3D.${ext}" "${MASTR3_SCENE}/dense-view-sparse/0/points3D.${ext}"
done

echo
echo "[INFO] MASt3R scene: $MASTR3_SCENE"
echo "[INFO] Free-gaussians dir: $FREE_GAUSSIANS_DIR"
echo "[INFO] Free-gaussians config: $CONFIG"
echo "[INFO] Iteration: $ITERATION"
echo "[INFO] Resolution factor: $RESOLUTION"
echo "[INFO] First-stage point cloud source: $SOURCE_POINT_CLOUD_DIR"
echo "[INFO] Plane merge device: $MERGE_DEVICE"
echo "[INFO] Plane merge resolution scale: $MERGE_RESOLUTION_SCALE"

run_cmd rm -rf "${MASTR3_SCENE}/render-dense-train-views"
run_cmd rm -rf "$PLANE_ROOT_PATH"
run_cmd rm -rf "$TRAIN_DIR"
run_cmd rm -rf "$TETRA_MESHES_DIR"

if [[ "$SOURCE_POINT_CLOUD_DIR" == "$POINT_CLOUD_CHART_DIR" ]]; then
  run_cmd rm -rf "$POINT_CLOUD_DIR"
  run_cmd cp -a "$POINT_CLOUD_CHART_DIR" "$POINT_CLOUD_DIR"
fi

run_cmd pixi run python 2d-gaussian-splatting/render_dense_views.py \
  --source_path "$MASTR3_SCENE" \
  --model_path "$FREE_GAUSSIANS_DIR" \
  --iteration "$ITERATION"

run_cmd pixi run python 2d-gaussian-splatting/guidance/dense_dn_util.py \
  --source_path "$MASTR3_SCENE" \
  --model_path "$FREE_GAUSSIANS_DIR" \
  --iteration "$ITERATION"

run_cmd pixi run python 2d-gaussian-splatting/planes/plane_excavator.py \
  --plane_root_path "$PLANE_ROOT_PATH"

run_cmd pixi run python scripts/plane_refine_depth.py \
  --source_path "$MASTR3_SCENE" \
  --plane_root_path "$PLANE_ROOT_PATH" \
  --pnts_path "${MASTR3_SCENE}/chart_pcd.ply" \
  --resolution "$RESOLUTION" \
  --merge_resolution_scale "$MERGE_RESOLUTION_SCALE" \
  --merge_device "$MERGE_DEVICE"

if [[ ! -d "$POINT_CLOUD_CHART_DIR" ]]; then
  run_cmd mv "$POINT_CLOUD_DIR" "$POINT_CLOUD_CHART_DIR"
else
  run_cmd rm -rf "$POINT_CLOUD_DIR"
fi

run_cmd pixi run python scripts/refine_free_gaussians.py \
  --mast3r_scene "$MASTR3_SCENE" \
  --output_path "$FREE_GAUSSIANS_DIR" \
  --config "$CONFIG" \
  --resolution "$RESOLUTION" \
  --refine_depth_path "$PLANE_ROOT_PATH"

run_cmd pixi run python 2d-gaussian-splatting/render_multires.py \
  --source_path "$MASTR3_SCENE" \
  --model_path "$FREE_GAUSSIANS_DIR" \
  --skip_test \
  --skip_mesh \
  --render_all_img \
  --use_default_output_dir

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
echo "[DONE] Dense-view continuation complete."
echo "[DONE] Final Gaussian: ${FREE_GAUSSIANS_DIR}/point_cloud/iteration_${ITERATION}/point_cloud.ply"
echo "[DONE] Chart-view backup: ${POINT_CLOUD_CHART_DIR}/iteration_${ITERATION}/point_cloud.ply"
