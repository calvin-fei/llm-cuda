import torch

from .extension import get_cuda_extension_load_error, load_cuda_extension


def _torch_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    var = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(var + eps) * weight


def cuda_rms_norm_extension(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor | None:
    if not x.is_cuda or not weight.is_cuda:
        return None

    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return None

    ext = load_cuda_extension(verbose=False)
    if ext is None:
        return None

    if not hasattr(ext, "rms_norm_forward"):
        return None

    try:
        return ext.rms_norm_forward(x, weight, float(eps))
    except Exception:
        return None


def cuda_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    out = cuda_rms_norm_extension(x, weight, eps)
    if out is not None:
        return out
    return _torch_rms_norm(x, weight, eps)


def cuda_extension_status() -> str:
    ext = load_cuda_extension(verbose=False)
    if ext is not None:
        return "loaded"

    error = get_cuda_extension_load_error()
    if error is None:
        return "not_loaded"
    return f"failed: {error}"
