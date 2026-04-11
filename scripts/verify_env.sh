#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
import pathlib
import sys

repo_root = pathlib.Path.cwd()
sys.path.insert(0, str(repo_root / "mast3r/dust3r/croco/models/curope"))

import torch
import torchvision
import torchaudio
import xformers  # noqa: F401
import open3d  # noqa: F401
import roma  # noqa: F401
import diff_surfel_rasterization  # noqa: F401
from simple_knn._C import distCUDA2  # noqa: F401
from tetranerf.utils.extension import cpp  # noqa: F401
import pytorch3d  # noqa: F401
from pytorch3d.ops import knn_points  # noqa: F401
import segment_anything  # noqa: F401
import asmk  # noqa: F401
import asmk.hamming  # noqa: F401
import curope  # noqa: F401

print(f"python={sys.version.split()[0]}")
print(f"torch={torch.__version__}")
print(f"torchvision={torchvision.__version__}")
print(f"torchaudio={torchaudio.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    caps = [torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())]
    print(f"cuda_caps={caps}")

try:
    import detectron2  # noqa: F401
except Exception as exc:
    print(f"detectron2=optional-missing ({exc.__class__.__name__}: {exc})")
else:
    print("detectron2=installed")

print("core_imports=ok")
PY
