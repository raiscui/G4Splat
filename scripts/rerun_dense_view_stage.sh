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
  --tetra-config <name>            Tetra extraction config. Default: default
  --tetra-downsample-ratio <f>     Tetra downsample ratio. Default: 0.25
  --visible-threshold <f>          dense depth visibility threshold. Default: YAML visible-threshold or 0.99
  --max-visible-rgb-error <f>      dense depth RGB error gate. Default: YAML max-visible-rgb-error or unset
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

run_dense_dn_util() {
  if [[ -n "$MAX_VISIBLE_RGB_ERROR" ]]; then
    run_cmd pixi run python 2d-gaussian-splatting/guidance/dense_dn_util.py \
      --source_path "$MASTR3_SCENE" \
      --model_path "$FREE_GAUSSIANS_DIR" \
      --iteration "$ITERATION" \
      --visible-threshold "$VISIBLE_THRESHOLD" \
      --max-visible-rgb-error "$MAX_VISIBLE_RGB_ERROR"
  else
    run_cmd pixi run python 2d-gaussian-splatting/guidance/dense_dn_util.py \
      --source_path "$MASTR3_SCENE" \
      --model_path "$FREE_GAUSSIANS_DIR" \
      --iteration "$ITERATION" \
      --visible-threshold "$VISIBLE_THRESHOLD"
  fi
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

load_visible_threshold_from_config() {
  local config_path="$1"
  local value
  value="$(sed -nE 's/^[[:space:]]*visible[-_]threshold[[:space:]]*:[[:space:]]*([^#[:space:]]+).*/\1/p' "$config_path" | head -n 1)"
  if [[ -n "$value" ]]; then
    printf '%s\n' "$value"
  else
    printf '0.99\n'
  fi
}

load_max_visible_rgb_error_from_config() {
  local config_path="$1"
  sed -nE 's/^[[:space:]]*max[-_]visible[-_]rgb[-_]error[[:space:]]*:[[:space:]]*([^#[:space:]]+).*/\1/p' "$config_path" | head -n 1
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
TETRA_CONFIG="default"
TETRA_DOWNSAMPLE_RATIO="0.25"
VISIBLE_THRESHOLD=""
MAX_VISIBLE_RGB_ERROR=""
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
    --backup-dir)
      BACKUP_DIR="$2"
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
    --visible-threshold)
      VISIBLE_THRESHOLD="$2"
      shift 2
      ;;
    --max-visible-rgb-error)
      MAX_VISIBLE_RGB_ERROR="$2"
      shift 2
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
CONFIG_PATH="configs/free_gaussians_refinement/${CONFIG}.yaml"
require_file "$CONFIG_PATH"

if [[ -z "$VISIBLE_THRESHOLD" ]]; then
  VISIBLE_THRESHOLD="$(load_visible_threshold_from_config "$CONFIG_PATH")"
fi
if [[ -z "$MAX_VISIBLE_RGB_ERROR" ]]; then
  MAX_VISIBLE_RGB_ERROR="$(load_max_visible_rgb_error_from_config "$CONFIG_PATH")"
fi

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
echo "[INFO] Dense visible threshold: $VISIBLE_THRESHOLD"
if [[ -n "$MAX_VISIBLE_RGB_ERROR" ]]; then
  echo "[INFO] Dense max visible RGB error: $MAX_VISIBLE_RGB_ERROR"
fi

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

run_dense_dn_util

run_cmd pixi run python 2d-gaussian-splatting/planes/plane_excavator.py \
  --plane_root_path "${MASTR3_SCENE}/plane-refine-depths"

run_cmd pixi run python scripts/plane_refine_depth.py \
  --source_path "$MASTR3_SCENE" \
  --plane_root_path "${MASTR3_SCENE}/plane-refine-depths" \
  --pnts_path "${MASTR3_SCENE}/chart_pcd.ply"

run_cmd pixi run python scripts/refine_free_gaussians.py \
  --mast3r_scene "$MASTR3_SCENE" \
  --output_path "$FREE_GAUSSIANS_DIR" \
  --config "$CONFIG" \
  --refine_depth_path "${MASTR3_SCENE}/plane-refine-depths"

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
echo "[DONE] Dense-view stage rerun complete."
