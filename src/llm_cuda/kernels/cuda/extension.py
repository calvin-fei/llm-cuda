from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_EXTENSION = None
_LOAD_ATTEMPTED = False
_LOAD_ERROR: str | None = None


def load_cuda_extension(verbose: bool = False):
    global _EXTENSION, _LOAD_ATTEMPTED, _LOAD_ERROR

    if _LOAD_ATTEMPTED:
        return _EXTENSION

    _LOAD_ATTEMPTED = True

    if not torch.cuda.is_available():
        _LOAD_ERROR = "CUDA is not available"
        return None

    try:
        root = Path(__file__).resolve().parent
        sources = [
            str(root / "csrc" / "rmsnorm_extension.cpp"),
            str(root / "csrc" / "rmsnorm_kernel.cu"),
            str(root / "csrc" / "swiglu_kernel.cu"),
            str(root / "csrc" / "attention_kernel.cu"),
        ]

        _EXTENSION = load(
            name="llm_cuda_kernels_cuda_ext",
            sources=sources,
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=verbose,
        )
    except Exception as exc:  # pragma: no cover - build environment dependent
        _LOAD_ERROR = str(exc)
        _EXTENSION = None

    return _EXTENSION


def get_cuda_extension_load_error() -> str | None:
    return _LOAD_ERROR
