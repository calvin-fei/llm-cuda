import torch

from llm_cuda.kernels.triton.rope import can_use_triton_rope, triton_apply_rope


def build_inv_freq(
    head_dim: int,
    base: float,
    device: torch.device,
) -> torch.Tensor:
    """Compute the inverse frequency vector for RoPE.

    This is a constant per model configuration and should be cached (e.g.
    as a module buffer) to avoid recomputing it on every forward step.

    Args:
        head_dim: Attention head dimension.
        base:     RoPE base frequency (``rope_theta``).
        device:   Target device.

    Returns:
        Float32 tensor of shape ``[head_dim // 2]``.
    """
    return 1.0 / (
        base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    base: float,
    device: torch.device,
    dtype: torch.dtype,
    position_offset: int = 0,
    inv_freq: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build cosine/sine tables for rotary positional embeddings.

    Args:
        seq_len:         Number of query/key positions to generate.
        head_dim:        Attention head dimension.
        base:            RoPE base frequency (``rope_theta``).
        device:          Target device.
        dtype:           Output dtype for cos/sin tables.
        position_offset: Starting position index (used during KV-cache decode).
        inv_freq:        Optional pre-computed inverse frequency vector
                         ``[head_dim // 2]``.  Pass the cached buffer from
                         :func:`build_inv_freq` to skip recomputing it every
                         step.  Moved to the correct device automatically if
                         needed.

    Returns:
        Tuple ``(cos, sin)`` each of shape ``[seq_len, head_dim]``.
    """
    if inv_freq is None:
        inv_freq = build_inv_freq(head_dim, base, device)
    elif inv_freq.device != device:
        inv_freq = inv_freq.to(device=device)

    positions = torch.arange(
        position_offset,
        position_offset + seq_len,
        device=device,
        dtype=torch.float32,
    )
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype=dtype)
    sin = emb.sin().to(dtype=dtype)
    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary positional embeddings to query and key tensors.

    Uses the fused Triton kernel (single pass, no intermediate allocations)
    when available; falls back to element-wise PyTorch operations otherwise.

    Args:
        q:   ``[batch, n_heads,    seq, head_dim]``
        k:   ``[batch, n_kv_heads, seq, head_dim]``
        cos: ``[seq, head_dim]``
        sin: ``[seq, head_dim]``

    Returns:
        ``(q_out, k_out)`` with the same shapes and dtypes as inputs.
    """
    if can_use_triton_rope(q, cos, sin) and can_use_triton_rope(k, cos, sin):
        return triton_apply_rope(q, cos, sin), triton_apply_rope(k, cos, sin)

    # PyTorch fallback: cos/sin must be broadcast over batch and heads.
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_out = (q * cos) + (rotate_half(q) * sin)
    k_out = (k * cos) + (rotate_half(k) * sin)
    return q_out, k_out
