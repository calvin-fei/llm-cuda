import torch

from .extension import load_cuda_extension


def cuda_swiglu_extension(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor | None:
    if not gate.is_cuda or not up.is_cuda:
        return None

    if gate.shape != up.shape:
        return None

    if gate.dtype not in (torch.float16, torch.float32):
        return None

    ext = load_cuda_extension(verbose=False)
    if ext is None:
        return None

    if not hasattr(ext, "swiglu_forward"):
        return None

    try:
        return ext.swiglu_forward(gate, up)
    except Exception:
        return None
