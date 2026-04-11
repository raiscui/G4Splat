#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/build_env_common.sh"

resolve_build_jobs "install-xformers" "XFORMERS_RESERVED_CPUS" 4
ensure_cuda_home
use_system_compiler_if_available
maybe_set_blackwell_arch

force_reinstall="${XFORMERS_FORCE_REINSTALL:-0}"
if ! [[ "${force_reinstall}" =~ ^(0|1)$ ]]; then
  echo "[install-xformers] XFORMERS_FORCE_REINSTALL 只能是 0 或 1, 当前值: ${force_reinstall}" >&2
  exit 1
fi

if [ "${force_reinstall}" = "0" ] && python_import_ok "xformers"; then
  echo "[install-xformers] 检测到可导入的 xformers, 跳过重复安装"
  exit 0
fi

xformers_version="${XFORMERS_VERSION:-0.0.34}"
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

echo "[install-xformers] 即将执行: python -m pip install ${pip_install_args[*]} xformers==${xformers_version}"
python -m pip install "${pip_install_args[@]}" "xformers==${xformers_version}"
