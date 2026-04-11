#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/build_env_common.sh"

resolve_build_jobs "install-detectron2" "DETECTRON2_RESERVED_CPUS" 4
ensure_cuda_home
use_system_compiler_if_available
maybe_set_blackwell_arch

force_reinstall="${DETECTRON2_FORCE_REINSTALL:-0}"
if ! [[ "${force_reinstall}" =~ ^(0|1)$ ]]; then
  echo "[install-detectron2] DETECTRON2_FORCE_REINSTALL 只能是 0 或 1, 当前值: ${force_reinstall}" >&2
  exit 1
fi

if [ "${force_reinstall}" = "0" ] && python_import_ok "detectron2"; then
  echo "[install-detectron2] 检测到可导入的 detectron2, 跳过重复安装"
  exit 0
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

echo "[install-detectron2] 即将执行源码安装"
MAX_JOBS="${BUILD_JOBS}" python -m pip install "${pip_install_args[@]}" "git+https://github.com/facebookresearch/detectron2.git"
