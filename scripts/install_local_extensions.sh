#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/build_env_common.sh"

resolve_build_jobs "install-local-extensions" "LOCAL_EXT_RESERVED_CPUS" 4
ensure_cuda_home
use_system_compiler_if_available
maybe_set_blackwell_arch

echo "[install-local-extensions] 安装 diff-surfel-rasterization"
if python_import_ok "diff_surfel_rasterization"; then
  echo "[install-local-extensions] diff_surfel_rasterization 已可导入, 跳过"
else
  python -m pip install --no-build-isolation --no-deps -e ./2d-gaussian-splatting/submodules/diff-surfel-rasterization
fi

echo "[install-local-extensions] 安装 simple-knn"
if python - <<'PY' >/dev/null 2>&1
import sys
try:
    import torch  # noqa: F401
    from simple_knn._C import distCUDA2  # noqa: F401
except Exception:
    sys.exit(1)
PY
then
  echo "[install-local-extensions] simple_knn 已可导入, 跳过"
else
  python -m pip install --no-build-isolation --no-deps -e ./2d-gaussian-splatting/submodules/simple-knn
fi

echo "[install-local-extensions] 编译 tetra-triangulation"
if python - <<'PY' >/dev/null 2>&1
import sys
try:
    from tetranerf.utils.extension import cpp  # noqa: F401
except Exception:
    sys.exit(1)
PY
then
  echo "[install-local-extensions] tetranerf 扩展已可导入, 跳过"
else
  pushd ./2d-gaussian-splatting/submodules/tetra-triangulation >/dev/null
  cmake .
  cmake --build . -j "${BUILD_JOBS}"
  popd >/dev/null
  python -m pip install --no-build-isolation --no-deps -e ./2d-gaussian-splatting/submodules/tetra-triangulation
fi

echo "[install-local-extensions] 编译并安装 mast3r/asmk"
if python - <<'PY' >/dev/null 2>&1
import sys
try:
    import asmk  # noqa: F401
    import asmk.hamming  # noqa: F401
except Exception:
    sys.exit(1)
PY
then
  echo "[install-local-extensions] asmk 已可导入, 跳过"
else
  pushd ./mast3r/asmk/cython >/dev/null
  cythonize *.pyx
  popd >/dev/null
  python -m pip install --no-build-isolation --no-deps ./mast3r/asmk
fi

echo "[install-local-extensions] 编译 mast3r curope"
if python - <<'PY' >/dev/null 2>&1
import pathlib
import sys

repo_root = pathlib.Path.cwd()
sys.path.insert(0, str(repo_root / "mast3r/dust3r/croco/models/curope"))
try:
    import torch  # noqa: F401
    import curope  # noqa: F401
except Exception:
    sys.exit(1)
PY
then
  echo "[install-local-extensions] curope 已可导入, 跳过"
else
  pushd ./mast3r/dust3r/croco/models/curope >/dev/null
  MAX_JOBS="${BUILD_JOBS}" python setup.py build_ext --inplace
  popd >/dev/null
fi
