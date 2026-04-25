"""Triton fused rotary positional embedding (RoPE) kernel for LLM inference.

Replaces the PyTorch ``rotate_half`` + element-wise multiply + add sequence
with a single Triton kernel, eliminating intermediate allocations and extra
kernel launches.  Each program handles one ``(batch, head, seq_pos)`` triplet,
loading both halves of the head dimension in one pass.

Key properties:

* **Single-pass**: cos/sin lookup and the rotation are merged — no intermediate
  ``rotate_half`` tensor is ever materialised.
* **Half-load optimisation**: since ``build_rope_cache`` produces
  ``cos[:, :half] == cos[:, half:]`` (via ``cat(freqs, freqs)``), the kernel
  reads only the first half of the cos/sin tables, halving bandwidth for those
  arrays.
* **Float32 accumulation**: intermediate arithmetic is performed in float32
  regardless of the input dtype, matching the precision of the reference
  PyTorch path.
"""

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None

# Half-dimension bounds for BLOCK_HALF (must be a power of two).
_MIN_BLOCK_HALF: int = 8
_MAX_BLOCK_HALF: int = 128


if triton is not None:

    @triton.jit
    def _apply_rope_fwd_kernel(
        x_ptr,
        cos_ptr,
        sin_ptr,
        out_ptr,
        stride_xb,
        stride_xh,
        stride_xs,
        stride_xd,
        stride_cos_s,
        stride_cos_d,
        stride_sin_s,
        stride_sin_d,
        stride_ob,
        stride_oh,
        stride_os,
        stride_od,
        n_heads,
        seq_len,
        head_dim,
        BLOCK_HALF: tl.constexpr,
    ):
        """One Triton program per ``(bsz * n_heads * seq_len)`` position.

        Loads both halves of the head dimension, applies the RoPE rotation
        (equivalent to ``rotate_half`` + element-wise multiply), and stores
        the result.

        ``cos`` / ``sin`` are expected to have shape ``[seq_len, head_dim]``
        where ``cos[:, :half] == cos[:, half:]`` (produced by
        ``cat(freqs, freqs)``).  Only the first half is therefore read.

        The rotation formula per position ``s`` and head dimension ``d``:

        .. code-block:: none

            out[d]       = x[d]      * cos[d] - x[d+half] * sin[d]   # d < half
            out[d+half]  = x[d+half] * cos[d] + x[d]      * sin[d]   # d < half
        """
        pid = tl.program_id(0)
        seq_idx = pid % seq_len
        tmp = pid // seq_len
        head_idx = tmp % n_heads
        bsz_idx = tmp // n_heads

        half = head_dim // 2
        offs = tl.arange(0, BLOCK_HALF)
        mask = offs < half

        x_base = (
            x_ptr
            + bsz_idx * stride_xb
            + head_idx * stride_xh
            + seq_idx * stride_xs
        )
        cos_base = cos_ptr + seq_idx * stride_cos_s
        sin_base = sin_ptr + seq_idx * stride_sin_s
        out_base = (
            out_ptr
            + bsz_idx * stride_ob
            + head_idx * stride_oh
            + seq_idx * stride_os
        )

        # Load first and second halves of x.
        x1 = tl.load(x_base + offs * stride_xd, mask=mask, other=0.0)
        x2 = tl.load(x_base + (offs + half) * stride_xd, mask=mask, other=0.0)

        # Load cos/sin (first half only — second half is identical).
        cos_v = tl.load(cos_base + offs * stride_cos_d, mask=mask, other=1.0)
        sin_v = tl.load(sin_base + offs * stride_sin_d, mask=mask, other=0.0)

        # Apply rotation in float32 for numerical stability.
        x1_f = x1.to(tl.float32)
        x2_f = x2.to(tl.float32)
        cos_f = cos_v.to(tl.float32)
        sin_f = sin_v.to(tl.float32)

        out1 = x1_f * cos_f - x2_f * sin_f
        out2 = x2_f * cos_f + x1_f * sin_f

        orig_dtype = x1.dtype
        tl.store(out_base + offs * stride_od, out1.to(orig_dtype), mask=mask)
        tl.store(
            out_base + (offs + half) * stride_od,
            out2.to(orig_dtype),
            mask=mask,
        )


def can_use_triton_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> bool:
    """Return True when the fused Triton RoPE kernel can be applied."""
    if triton is None:
        return False
    if not (x.is_cuda and cos.is_cuda and sin.is_cuda):
        return False
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if x.dim() != 4 or cos.dim() != 2 or sin.dim() != 2:
        return False
    head_dim = x.shape[-1]
    if head_dim % 2 != 0:
        return False
    half = head_dim // 2
    if half > _MAX_BLOCK_HALF:
        return False
    if cos.shape != sin.shape:
        return False
    # cos/sin must cover exactly the sequence positions in x.
    if cos.shape != (x.shape[-2], head_dim):
        return False
    return True


def triton_apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary positional embeddings to *x* using a fused Triton kernel.

    Args:
        x:   ``[bsz, n_heads, seq_len, head_dim]`` — query or key tensor.
        cos: ``[seq_len, head_dim]`` — precomputed cosine table from
             :func:`~llm_cuda.models.llama3.rotary.build_rope_cache`.
        sin: ``[seq_len, head_dim]`` — precomputed sine table.

    Returns:
        Rotated tensor with the same shape and dtype as ``x``.

    ``cos`` and ``sin`` must have the same shape and strides.  Call
    :func:`can_use_triton_rope` to verify preconditions before calling
    this function.
    """
    assert triton is not None, "Triton is not available"

    bsz, n_heads, seq_len, head_dim = x.shape

    if not x.is_contiguous():
        x = x.contiguous()
    if not cos.is_contiguous():
        cos = cos.contiguous()
    if not sin.is_contiguous():
        sin = sin.contiguous()

    out = torch.empty_like(x)

    half = head_dim // 2
    # BLOCK_HALF: smallest power of 2 >= half, clamped.
    block_half = triton.next_power_of_2(half)
    block_half = min(max(block_half, _MIN_BLOCK_HALF), _MAX_BLOCK_HALF)

    # Number of warps: 1 warp per 32 elements, minimum 1.
    num_warps = max(1, block_half // 32)

    grid = (bsz * n_heads * seq_len,)
    _apply_rope_fwd_kernel[grid](
        x,
        cos,
        sin,
        out,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        cos.stride(0),
        cos.stride(1),
        sin.stride(0),
        sin.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        n_heads,
        seq_len,
        head_dim,
        BLOCK_HALF=block_half,
        num_warps=num_warps,
    )

    return out
