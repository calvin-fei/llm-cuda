"""Triton flash-decode attention kernel for LLM decoding workloads.

Designed specifically for the decode step where the query has a single token
(q_len == 1) attending to a growing KV cache.  Key properties:

* **GQA-native**: handles Grouped-Query Attention without expanding K/V heads.
  The mapping ``kv_head = q_head // (n_heads // n_kv_heads)`` is computed
  inside the kernel, avoiding the ``_repeat_kv`` tensor copy entirely.
* **Fused single-pass**: QK dot-products, online softmax, and V accumulation
  are merged into one kernel — no intermediate attention-score tensor is
  materialised.
* **Tiled KV access**: iterates over the KV cache in ``BLOCK_N`` tiles so that
  K and V blocks are reused from L2, reducing global memory pressure as the
  cache grows.
"""

import math

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _decode_attention_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        # Q / O strides — seq dimension is always 1 so we skip stride_m.
        stride_qb,
        stride_qh,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vn,
        stride_vd,
        stride_ob,
        stride_oh,
        stride_od,
        n_heads,
        n_kv_heads,
        kv_len,
        head_dim,
        scale,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """One Triton program per (batch, query-head) pair.

        Each program:
        1. Loads the Q vector ``[head_dim]`` for its (batch, head).
        2. Iterates over the KV cache in tiles of ``BLOCK_N`` positions.
        3. For every tile: computes QK dot products, updates the online-softmax
           running statistics (max ``m_i``, normaliser ``l_i``), then
           accumulates the weighted V contribution.
        4. Writes the normalised output vector.
        """
        pid = tl.program_id(0)
        bsz_idx = pid // n_heads
        head_idx = pid % n_heads

        # GQA: n_heads Q-heads share n_kv_heads KV-heads.
        # Heads 0..(kv_groups-1) → kv_head 0, heads kv_groups..2*kv_groups-1 → kv_head 1, …
        kv_groups = n_heads // n_kv_heads
        kv_head_idx = head_idx // kv_groups

        offs_d = tl.arange(0, BLOCK_D)
        d_mask = offs_d < head_dim

        # Load Q vector: q[bsz_idx, head_idx, 0, offs_d]
        q = tl.load(
            q_ptr + bsz_idx * stride_qb + head_idx * stride_qh + offs_d * stride_qd,
            mask=d_mask,
            other=0.0,
        )

        # Online-softmax state.
        m_i = -float("inf")
        l_i = 0.0
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

        for start_n in tl.range(0, kv_len, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            n_mask = offs_n < kv_len

            # Load K tile: k[bsz_idx, kv_head_idx, offs_n, offs_d]
            k_ptrs = (
                k_ptr
                + bsz_idx * stride_kb
                + kv_head_idx * stride_kh
                + offs_n[:, None] * stride_kn
                + offs_d[None, :] * stride_kd
            )
            k = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)

            # QK dot products: [BLOCK_N]
            qk = tl.sum(k * q[None, :], axis=1) * scale
            # Mask positions beyond the KV cache length.
            qk = tl.where(n_mask, qk, -float("inf"))

            # Online softmax: update running max and normaliser.
            m_ij = tl.max(qk, axis=0)
            p = tl.exp(qk - m_ij)
            l_ij = tl.sum(p, axis=0)

            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_new)
            beta = tl.exp(m_ij - m_new)

            # Load V tile: v[bsz_idx, kv_head_idx, offs_n, offs_d]
            v_ptrs = (
                v_ptr
                + bsz_idx * stride_vb
                + kv_head_idx * stride_vh
                + offs_n[:, None] * stride_vn
                + offs_d[None, :] * stride_vd
            )
            v = tl.load(v_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)

            # Accumulate weighted V.
            acc = acc * alpha + tl.sum(v * (p * beta)[:, None], axis=0)
            l_i = l_i * alpha + l_ij * beta
            m_i = m_new

        # Normalise and write output: o[bsz_idx, head_idx, 0, offs_d]
        out = acc / tl.maximum(l_i, 1e-9)
        tl.store(
            o_ptr + bsz_idx * stride_ob + head_idx * stride_oh + offs_d * stride_od,
            out,
            mask=d_mask,
        )


def can_use_triton_decode_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> bool:
    """Return True when the Triton flash-decode kernel can be applied."""
    if triton is None:
        return False
    if attention_mask is not None:
        return False
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        return False
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        return False
    # Decode step: exactly one query token.
    if q.shape[-2] != 1:
        return False
    # Batch and head-dim must be consistent.
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        return False
    if q.shape[-1] != k.shape[-1] or q.shape[-1] != v.shape[-1]:
        return False
    # head_dim cap: BLOCK_D is capped at 256.
    if q.shape[-1] > 256:
        return False
    # K/V shapes must match.
    if k.shape != v.shape:
        return False
    # Must have at least one KV position to attend to.
    if k.shape[-2] == 0:
        return False
    # GQA constraint: n_heads must be a multiple of n_kv_heads.
    n_heads = q.shape[1]
    n_kv_heads = k.shape[1]
    if n_kv_heads == 0 or n_heads % n_kv_heads != 0:
        return False
    return True


def triton_decode_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Flash-decode attention for a single query token.

    Args:
        q: ``[bsz, n_heads, 1, head_dim]`` — query tensor (decode step).
        k: ``[bsz, n_kv_heads, kv_len, head_dim]`` — key cache.
        v: ``[bsz, n_kv_heads, kv_len, head_dim]`` — value cache.

    Returns:
        Output tensor with the same shape as ``q``.

    The kernel handles Grouped-Query Attention natively; ``n_heads`` must be an
    integer multiple of ``n_kv_heads``.  Call ``can_use_triton_decode_attention``
    to verify preconditions before calling this function.
    """
    assert triton is not None, "Triton is not available"

    bsz, n_heads, _, head_dim = q.shape
    n_kv_heads = k.shape[1]
    kv_len = k.shape[2]
    scale = 1.0 / math.sqrt(head_dim)

    if not q.is_contiguous():
        q = q.contiguous()
    if not k.is_contiguous():
        k = k.contiguous()
    if not v.is_contiguous():
        v = v.contiguous()

    out = torch.empty_like(q)

    # BLOCK_D: smallest power of 2 >= head_dim, capped at 256.
    block_d = triton.next_power_of_2(head_dim)
    block_d = min(max(block_d, 16), 256)

    # BLOCK_N: larger tiles improve bandwidth for long KV sequences.
    if kv_len >= 2048:
        block_n = 128
    elif kv_len >= 512:
        block_n = 64
    else:
        block_n = 32

    num_warps = 4 if block_n * block_d >= 4096 else 2

    grid = (bsz * n_heads,)
    _decode_attention_kernel[grid](
        q,
        k,
        v,
        out,
        # Q strides (skip seq stride — always 1 token at position 0).
        q.stride(0),
        q.stride(1),
        q.stride(3),
        # K strides.
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        # V strides.
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        # O strides (skip seq stride).
        out.stride(0),
        out.stride(1),
        out.stride(3),
        n_heads,
        n_kv_heads,
        kv_len,
        head_dim,
        scale,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=num_warps,
    )

    return out
