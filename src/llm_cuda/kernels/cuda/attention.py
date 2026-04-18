import torch

from .extension import load_cuda_extension


def cuda_causal_attention_extension(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor | None:
    if attention_mask is not None:
        return None

    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        return None

    if q.shape != k.shape or q.shape != v.shape:
        return None

    if q.dim() != 4:
        return None

    if q.dtype not in (torch.float16, torch.float32):
        return None

    ext = load_cuda_extension(verbose=False)
    if ext is None:
        return None

    if not hasattr(ext, "attention_forward"):
        return None

    try:
        return ext.attention_forward(q, k, v)
    except Exception:
        return None
