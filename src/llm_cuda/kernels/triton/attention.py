import math

import torch
import torch.nn.functional as F

from llm_cuda.kernels.cuda.attention import cuda_causal_attention_extension

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None


def _select_attention_block_sizes(seq_len: int, head_dim: int) -> tuple[int, int]:
    if triton is None:
        block_d = min(max(1 << (head_dim - 1).bit_length(), 16), 128)
    else:
        block_d = triton.next_power_of_2(head_dim)
        block_d = min(max(block_d, 16), 128)

    if seq_len >= 1024:
        block_n = 128
    elif seq_len >= 256:
        block_n = 64
    else:
        block_n = 32
    return block_n, block_d


def _select_attention_num_warps(head_dim: int, block_n: int) -> int:
    work = head_dim * block_n
    if work >= 8192:
        return 8
    if work >= 4096:
        return 4
    return 2


def _should_use_cuda_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> bool:
    if attention_mask is not None:
        return False
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        return False
    if q.shape != k.shape or q.shape != v.shape:
        return False
    if q.dtype not in (torch.float16, torch.bfloat16):
        return False

    bsz, n_heads, seq_len, _ = q.shape
    # Prefer highly optimized SDPA kernels for long context or high concurrency.
    return seq_len >= 512 or (bsz * n_heads >= 16 and seq_len >= 256)


def _cuda_sdpa_causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=True,
    )


if triton is not None:

    @triton.jit
    def _fused_causal_attn_fwd(
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        logsumexp_ptr,
        stride_qb,
        stride_qh,
        stride_qm,
        stride_qk,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_kk,
        stride_vb,
        stride_vh,
        stride_vn,
        stride_vk,
        stride_ob,
        stride_oh,
        stride_om,
        stride_ok,
        stride_lseb,
        stride_lseh,
        stride_lsem,
        n_heads,
        seq_len,
        head_dim,
        scale,
        BLOCK_N: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)

        offs_d = tl.arange(0, BLOCK_DMODEL)
        offs_n = tl.arange(0, BLOCK_N)

        batch_idx = pid_bh // n_heads
        head_idx = pid_bh % n_heads

        q_base = q_ptr + batch_idx * stride_qb + head_idx * stride_qh + pid_m * stride_qm
        q = tl.load(q_base + offs_d * stride_qk, mask=offs_d < head_dim, other=0.0)

        m_i = -float("inf")
        l_i = 0.0
        acc = tl.zeros((BLOCK_DMODEL,), dtype=tl.float32)

        for start_n in tl.range(0, seq_len, BLOCK_N):
            n_idx = start_n + offs_n
            n_mask = n_idx < seq_len

            k_ptrs = (
                k_ptr
                + batch_idx * stride_kb
                + head_idx * stride_kh
                + n_idx[:, None] * stride_kn
                + offs_d[None, :] * stride_kk
            )
            v_ptrs = (
                v_ptr
                + batch_idx * stride_vb
                + head_idx * stride_vh
                + n_idx[:, None] * stride_vn
                + offs_d[None, :] * stride_vk
            )

            k = tl.load(k_ptrs, mask=n_mask[:, None] & (offs_d[None, :] < head_dim), other=0.0)
            v = tl.load(v_ptrs, mask=n_mask[:, None] & (offs_d[None, :] < head_dim), other=0.0)

            qk = tl.sum(k * q[None, :], axis=1) * scale
            causal_mask = n_idx <= pid_m
            qk = tl.where(n_mask & causal_mask, qk, -float("inf"))

            m_ij = tl.max(qk, axis=0)
            p = tl.exp(qk - m_ij)
            l_ij = tl.sum(p, axis=0)

            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_new)
            beta = tl.exp(m_ij - m_new)

            l_i = l_i * alpha + l_ij * beta
            p_scaled = p * beta

            acc = acc * alpha + tl.sum(v * p_scaled[:, None], axis=0)
            m_i = m_new

        l_i = tl.maximum(l_i, 1e-9)
        out = acc / l_i

        o_base = o_ptr + batch_idx * stride_ob + head_idx * stride_oh + pid_m * stride_om
        tl.store(o_base + offs_d * stride_ok, out, mask=offs_d < head_dim)

        # Save logsumexp = m_i + log(l_i) for the backward pass.
        lse = m_i + tl.log(l_i)
        lse_ptr = logsumexp_ptr + batch_idx * stride_lseb + head_idx * stride_lseh + pid_m * stride_lsem
        tl.store(lse_ptr, lse)

    @triton.jit
    def _fused_causal_attn_bwd(
        q_ptr,
        k_ptr,
        v_ptr,
        do_ptr,
        dq_ptr,
        dk_ptr,
        dv_ptr,
        logsumexp_ptr,
        delta_ptr,
        stride_qb,
        stride_qh,
        stride_qm,
        stride_qk,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_kk,
        stride_vb,
        stride_vh,
        stride_vn,
        stride_vk,
        stride_dob,
        stride_doh,
        stride_dom,
        stride_dok,
        stride_dqb,
        stride_dqh,
        stride_dqm,
        stride_dqk,
        stride_dkb,
        stride_dkh,
        stride_dkn,
        stride_dkk,
        stride_dvb,
        stride_dvh,
        stride_dvn,
        stride_dvk,
        stride_lseb,
        stride_lseh,
        stride_lsem,
        stride_deltab,
        stride_deltah,
        stride_deltam,
        n_heads,
        seq_len,
        head_dim,
        scale,
        BLOCK_N: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)

        offs_d = tl.arange(0, BLOCK_DMODEL)
        offs_n = tl.arange(0, BLOCK_N)

        batch_idx = pid_bh // n_heads
        head_idx = pid_bh % n_heads

        q_base = q_ptr + batch_idx * stride_qb + head_idx * stride_qh + pid_m * stride_qm
        do_base = do_ptr + batch_idx * stride_dob + head_idx * stride_doh + pid_m * stride_dom
        q = tl.load(q_base + offs_d * stride_qk, mask=offs_d < head_dim, other=0.0)
        do = tl.load(do_base + offs_d * stride_dok, mask=offs_d < head_dim, other=0.0)

        # Load saved logsumexp and precomputed delta D = rowsum(dO * O).
        lse = tl.load(logsumexp_ptr + batch_idx * stride_lseb + head_idx * stride_lseh + pid_m * stride_lsem)
        delta = tl.load(delta_ptr + batch_idx * stride_deltab + head_idx * stride_deltah + pid_m * stride_deltam)

        dq_acc = tl.zeros((BLOCK_DMODEL,), dtype=tl.float32)

        for start_n in tl.range(0, seq_len, BLOCK_N):
            n_idx = start_n + offs_n
            causal = n_idx <= pid_m
            n_mask = (n_idx < seq_len) & causal

            k_ptrs = (
                k_ptr
                + batch_idx * stride_kb
                + head_idx * stride_kh
                + n_idx[:, None] * stride_kn
                + offs_d[None, :] * stride_kk
            )
            v_ptrs = (
                v_ptr
                + batch_idx * stride_vb
                + head_idx * stride_vh
                + n_idx[:, None] * stride_vn
                + offs_d[None, :] * stride_vk
            )

            k = tl.load(k_ptrs, mask=n_mask[:, None] & (offs_d[None, :] < head_dim), other=0.0)
            v = tl.load(v_ptrs, mask=n_mask[:, None] & (offs_d[None, :] < head_dim), other=0.0)

            qk = tl.sum(k * q[None, :], axis=1) * scale
            qk = tl.where(n_mask, qk, -float("inf"))

            # Reuse the saved logsumexp to recover softmax probabilities directly,
            # avoiding separate passes to recompute m_i and l_i.
            p = tl.exp(qk - lse)
            p = tl.where(n_mask, p, 0.0)

            dp = tl.sum(v * do[None, :], axis=1)
            dp = tl.where(n_mask, dp, 0.0)

            # delta = rowsum(dO * O) = sum_j(p_j * dp_j) (precomputed in Python).
            ds = p * (dp - delta)
            ds = tl.where(n_mask, ds, 0.0)

            dq_acc += tl.sum(ds[:, None] * k, axis=0) * scale

            dk_ptrs = (
                dk_ptr
                + batch_idx * stride_dkb
                + head_idx * stride_dkh
                + n_idx[:, None] * stride_dkn
                + offs_d[None, :] * stride_dkk
            )
            dv_ptrs = (
                dv_ptr
                + batch_idx * stride_dvb
                + head_idx * stride_dvh
                + n_idx[:, None] * stride_dvn
                + offs_d[None, :] * stride_dvk
            )

            dk_update = ds[:, None] * q[None, :] * scale
            dv_update = p[:, None] * do[None, :]
            mask_2d = n_mask[:, None] & (offs_d[None, :] < head_dim)
            tl.atomic_add(dk_ptrs, dk_update, mask=mask_2d)
            tl.atomic_add(dv_ptrs, dv_update, mask=mask_2d)

        dq_base = dq_ptr + batch_idx * stride_dqb + head_idx * stride_dqh + pid_m * stride_dqm
        tl.store(dq_base + offs_d * stride_dqk, dq_acc, mask=offs_d < head_dim)


class _FusedCausalAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        bsz, n_heads, seq_len, head_dim = q.shape
        scale = 1.0 / math.sqrt(head_dim)

        out = torch.empty_like(q)
        # logsumexp is kept in float32 regardless of input dtype to preserve
        # numerical precision during the backward pass: it stores m_i + log(l_i)
        # which is used to recover exact softmax probabilities via exp(qk - lse).
        logsumexp = torch.empty(bsz, n_heads, seq_len, dtype=torch.float32, device=q.device)

        block_n, block_d = _select_attention_block_sizes(seq_len=seq_len, head_dim=head_dim)
        num_warps = _select_attention_num_warps(head_dim=head_dim, block_n=block_n)

        grid = (seq_len, bsz * n_heads)
        _fused_causal_attn_fwd[grid](
            q,
            k,
            v,
            out,
            logsumexp,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            logsumexp.stride(0),
            logsumexp.stride(1),
            logsumexp.stride(2),
            n_heads,
            seq_len,
            head_dim,
            scale,
            BLOCK_N=block_n,
            BLOCK_DMODEL=block_d,
            num_warps=num_warps,
        )

        ctx.save_for_backward(q, k, v, out, logsumexp)
        ctx.scale = scale
        ctx.block_n = block_n
        ctx.block_d = block_d
        ctx.num_warps = num_warps
        return out

    @staticmethod
    def backward(ctx, do: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v, out, logsumexp = ctx.saved_tensors
        bsz, n_heads, seq_len, head_dim = q.shape

        if not do.is_contiguous():
            do = do.contiguous()

        # Precompute delta D = rowsum(dO * O) — equivalent to sum_j(p_j * dp_j).
        # Derivation: D = sum_d(dO[d] * O[d])
        #               = sum_d(dO[d] * sum_j(p_j * V[j][d]))
        #               = sum_j(p_j * sum_d(dO[d] * V[j][d]))
        #               = sum_j(p_j * dp_j)
        # This collapses what was previously a dedicated kernel pass into a single
        # Python-side reduction, reducing the kernel from 4 passes to 1.
        delta = (do.float() * out.float()).sum(dim=-1).contiguous()

        dq = torch.empty_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        grid = (seq_len, bsz * n_heads)
        _fused_causal_attn_bwd[grid](
            q,
            k,
            v,
            do,
            dq,
            dk,
            dv,
            logsumexp,
            delta,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            do.stride(0),
            do.stride(1),
            do.stride(2),
            do.stride(3),
            dq.stride(0),
            dq.stride(1),
            dq.stride(2),
            dq.stride(3),
            dk.stride(0),
            dk.stride(1),
            dk.stride(2),
            dk.stride(3),
            dv.stride(0),
            dv.stride(1),
            dv.stride(2),
            dv.stride(3),
            logsumexp.stride(0),
            logsumexp.stride(1),
            logsumexp.stride(2),
            delta.stride(0),
            delta.stride(1),
            delta.stride(2),
            n_heads,
            seq_len,
            head_dim,
            ctx.scale,
            BLOCK_N=ctx.block_n,
            BLOCK_DMODEL=ctx.block_d,
            num_warps=ctx.num_warps,
        )
        return dq, dk, dv


def _torch_sdpa_fallback(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    seq_len = q.shape[-2]
    causal_mask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=q.device, dtype=q.dtype),
        diagonal=1,
    )
    attn_scores = attn_scores + causal_mask

    if attention_mask is not None:
        attn_scores = attn_scores + attention_mask

    attn_probs = torch.softmax(attn_scores, dim=-1)
    return torch.matmul(attn_probs, v)


def can_use_triton_fused_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> bool:
    if triton is None:
        return False
    if attention_mask is not None:
        return False
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        return False
    if q.shape != k.shape or q.shape != v.shape:
        return False
    if q.dtype not in (torch.float16, torch.bfloat16):
        return False
    if q.shape[-1] > 128:
        return False
    return True


def triton_fused_causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute causal attention for tensors of shape [batch, heads, seq, head_dim].
    Falls back to a torch implementation when Triton kernel constraints are not met.
    """
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, v must have matching shapes [batch, heads, seq, head_dim]")

    out_cuda = cuda_causal_attention_extension(q, k, v, attention_mask=attention_mask)
    if out_cuda is not None:
        return out_cuda

    if _should_use_cuda_sdpa(q, k, v, attention_mask=attention_mask):
        return _cuda_sdpa_causal_attention(q, k, v)

    scale = 1.0 / math.sqrt(q.shape[-1])

    if not can_use_triton_fused_attention(q, k, v, attention_mask=attention_mask):
        return _torch_sdpa_fallback(q, k, v, scale, attention_mask)

    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

    bsz, n_heads, seq_len, head_dim = q.shape
    if head_dim > 128:
        return _torch_sdpa_fallback(q, k, v, scale, attention_mask)

    return _FusedCausalAttentionFunction.apply(q, k, v)

