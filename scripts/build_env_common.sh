#!/usr/bin/env bash
set -euo pipefail

prepend_path_once() {
  local var_name="$1"
  local path_to_add="$2"
  local current_value="${!var_name:-}"

  case ":${current_value}:" in
    *":${path_to_add}:"*) ;;
    *)
      if [ -n "${current_value}" ]; then
        export "${var_name}=${path_to_add}:${current_value}"
      else
        export "${var_name}=${path_to_add}"
      fi
      ;;
  esac
}

ensure_cuda_home() {
  if [ -n "${CUDA_HOME:-}" ] && [ -x "${CUDA_HOME}/bin/nvcc" ]; then
    :
  elif [ -x /usr/local/cuda/bin/nvcc ]; then
    export CUDA_HOME="/usr/local/cuda"
  else
    local candidate=""
    candidate="$(find /usr/local -maxdepth 1 -type d -name 'cuda-*' | sort -V | tail -n 1 || true)"
    if [ -n "${candidate}" ] && [ -x "${candidate}/bin/nvcc" ]; then
      export CUDA_HOME="${candidate}"
    fi
  fi

  if [ -z "${CUDA_HOME:-}" ] || [ ! -x "${CUDA_HOME}/bin/nvcc" ]; then
    echo "[build-env] 未找到可用 CUDA toolkit. 期望存在 /usr/local/cuda/bin/nvcc 或显式设置有效的 CUDA_HOME." >&2
    return 1
  fi

  prepend_path_once PATH "${CUDA_HOME}/bin"
  if [ -d "${CUDA_HOME}/lib64" ]; then
    prepend_path_once LD_LIBRARY_PATH "${CUDA_HOME}/lib64"
  fi
  echo "[build-env] CUDA_HOME=${CUDA_HOME}"
}

resolve_build_jobs() {
  local log_prefix="$1"
  local reserve_var_name="$2"
  local reserve_default="${3:-4}"
  local reserve_value=""

  if [ -n "${MAX_JOBS:-}" ]; then
    BUILD_JOBS="${MAX_JOBS}"
    echo "[${log_prefix}] 使用外部指定的 MAX_JOBS=${BUILD_JOBS}"
  else
    reserve_value="${!reserve_var_name:-${reserve_default}}"
    if ! [[ "${reserve_value}" =~ ^[0-9]+$ ]]; then
      echo "[${log_prefix}] ${reserve_var_name} 必须是非负整数, 当前值: ${reserve_value}" >&2
      return 1
    fi

    local cpu_total
    cpu_total="$(nproc)"
    if [ "${cpu_total}" -gt "${reserve_value}" ]; then
      BUILD_JOBS="$((cpu_total - reserve_value))"
    else
      BUILD_JOBS=1
    fi
    echo "[${log_prefix}] cpu_total=${cpu_total}, reserved=${reserve_value}, MAX_JOBS=${BUILD_JOBS}"
  fi

  export BUILD_JOBS
  export MAX_JOBS="${BUILD_JOBS}"
  export CMAKE_BUILD_PARALLEL_LEVEL="${BUILD_JOBS}"
}

use_system_compiler_if_available() {
  if [ -x /usr/bin/gcc ] && [ -x /usr/bin/g++ ]; then
    export CC="${CC:-/usr/bin/gcc}"
    export CXX="${CXX:-/usr/bin/g++}"
    export CUDAHOSTCXX="${CUDAHOSTCXX:-/usr/bin/g++}"
    echo "[build-env] 使用 system gcc/g++ 作为 host compiler"
  fi
}

maybe_set_blackwell_arch() {
  if [ -n "${TORCH_CUDA_ARCH_LIST:-}" ]; then
    echo "[build-env] 使用现有 TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
    return 0
  fi

  local detected_arch=""
  detected_arch="$(
    python - <<'PY'
import re
import sys

try:
    import torch
except Exception:
    sys.exit(0)

if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
    sys.exit(0)

caps = sorted(
    {
        f"{major}.{minor}"
        for major, minor in (
            torch.cuda.get_device_capability(index)
            for index in range(torch.cuda.device_count())
        )
    }
)
if caps != ["12.0"]:
    sys.exit(0)

torch_match = re.match(r"^(\d+)\.(\d+)", torch.__version__)
cuda_match = re.match(r"^(\d+)\.(\d+)", torch.version.cuda or "")
if not torch_match or not cuda_match:
    sys.exit(0)

torch_version = tuple(int(part) for part in torch_match.groups())
cuda_version = tuple(int(part) for part in cuda_match.groups())
if torch_version >= (2, 7) and cuda_version >= (12, 8):
    print("12.0")
PY
  )"

  if [ -n "${detected_arch}" ]; then
    export TORCH_CUDA_ARCH_LIST="${detected_arch}"
    echo "[build-env] 自动设置 TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
  fi
}

python_import_ok() {
  local module_name="$1"
  python - <<PY >/dev/null 2>&1
import importlib
import sys

try:
    importlib.import_module("${module_name}")
except Exception:
    sys.exit(1)
PY
}
