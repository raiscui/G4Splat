#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/build_env_common.sh"

repo_dir="${PYTORCH3D_REPO_DIR:-.pixi/build-src/pytorch3d}"
repo_url="${PYTORCH3D_REPO_URL:-https://github.com/facebookresearch/pytorch3d.git}"
repo_ref="${PYTORCH3D_GIT_REF:-stable}"
force_reinstall="${PYTORCH3D_FORCE_REINSTALL:-0}"

resolve_build_jobs "install-pytorch3d" "PYTORCH3D_RESERVED_CPUS" 4
ensure_cuda_home
use_system_compiler_if_available
maybe_set_blackwell_arch

if ! [[ "${force_reinstall}" =~ ^(0|1)$ ]]; then
  echo "[install-pytorch3d] PYTORCH3D_FORCE_REINSTALL 只能是 0 或 1, 当前值: ${force_reinstall}" >&2
  exit 1
fi

if [ "${force_reinstall}" = "0" ] && python - <<'PY' >/dev/null 2>&1
import sys

try:
    import pytorch3d
    from pytorch3d.ops import knn_points  # noqa: F401
except Exception:
    sys.exit(1)
PY
then
  echo "[install-pytorch3d] 检测到可导入的 pytorch3d, 跳过重复安装"
  exit 0
fi

clone_repo() {
  echo "[install-pytorch3d] 使用浅克隆拉取 pytorch3d (${repo_ref})"
  mkdir -p "$(dirname "${repo_dir}")"
  rm -rf "${repo_dir}"
  git clone \
    --branch "${repo_ref}" \
    --single-branch \
    --depth 1 \
    --filter=blob:none \
    "${repo_url}" \
    "${repo_dir}"
}

if [ -d "${repo_dir}/.git" ]; then
  current_branch="$(git -C "${repo_dir}" symbolic-ref --short -q HEAD 2>/dev/null || true)"
  if [ "${current_branch:-}" != "${repo_ref}" ] && [ -z "$(git -C "${repo_dir}" status --porcelain 2>/dev/null || true)" ]; then
    clone_repo
  else
    echo "[install-pytorch3d] 复用现有仓库 ${repo_dir}"
  fi
else
  clone_repo
fi

pip_install_args=(
  --no-build-isolation
  --no-deps
)
if [ "${force_reinstall}" = "1" ]; then
  pip_install_args+=(
    --force-reinstall
    --no-cache-dir
  )
fi

echo "[install-pytorch3d] 即将执行: python -m pip install ${pip_install_args[*]} ./${repo_dir}"
MAX_JOBS="${BUILD_JOBS}" \
CMAKE_BUILD_PARALLEL_LEVEL="${BUILD_JOBS}" \
python -m pip install "${pip_install_args[@]}" "./${repo_dir}"
