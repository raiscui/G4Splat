# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

from setuptools import setup
from torch import cuda
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

def _normalize_arch_token(raw_arch: str) -> str:
    arch = raw_arch.strip()
    if arch.startswith("sm_"):
        arch = arch[3:]
    elif arch.startswith("compute_"):
        arch = arch[8:]
    if "." in arch:
        major, minor = arch.split(".", 1)
        arch = f"{major}{minor}"
    return arch


def _resolve_cuda_arch_flags():
    manual_arch_list = os.environ.get("CUROPE_CUDA_ARCH_LIST") or os.environ.get("TORCH_CUDA_ARCH_LIST")
    if manual_arch_list:
        flags = []
        for raw_arch in manual_arch_list.replace(";", " ").split():
            arch = _normalize_arch_token(raw_arch)
            if arch:
                flags.extend(["-gencode", f"arch=compute_{arch},code=sm_{arch}"])
        if flags:
            return flags
    return cuda.get_gencode_flags().replace("compute=", "arch=").split()


# compile for the requested CUDA architectures; default back to torch's full list only if no override exists.
all_cuda_archs = _resolve_cuda_arch_flags()
# alternatively, you can list cuda archs that you want, eg:
# all_cuda_archs = [
    # '-gencode', 'arch=compute_70,code=sm_70',
    # '-gencode', 'arch=compute_75,code=sm_75',
    # '-gencode', 'arch=compute_80,code=sm_80',
    # '-gencode', 'arch=compute_86,code=sm_86'
# ]

setup(
    name = 'curope',
    ext_modules = [
        CUDAExtension(
                name='curope',
                sources=[
                    "curope.cpp",
                    "kernels.cu",
                ],
                extra_compile_args = dict(
                    nvcc=['-O3','--ptxas-options=-v',"--use_fast_math"]+all_cuda_archs, 
                    cxx=['-O3'])
                )
    ],
    cmdclass = {
        'build_ext': BuildExtension
    })
