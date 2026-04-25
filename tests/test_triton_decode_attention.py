"""Tests for the Triton flash-decode attention kernel.

All correctness tests compare the kernel output against a plain-Python
reference that performs the equivalent MHA/GQA attention computation.
"""

import math

import pytest
import torch

from llm_cuda.kernels.triton.decode_attention import (
    can_use_triton_decode_attention,
    triton_decode_attention,
)


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------


_TEST_SEED: int = 42


def _reference_decode_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Pure-Python GQA decode attention reference.

    Args:
        q: ``[bsz, n_heads, 1, head_dim]``
        k: ``[bsz, n_kv_heads, kv_len, head_dim]``
        v: ``[bsz, n_kv_heads, kv_len, head_dim]``

    Returns:
        ``[bsz, n_heads, 1, head_dim]``
    """
    bsz, n_heads, _, head_dim = q.shape
    n_kv_heads = k.shape[1]
    kv_groups = n_heads // n_kv_heads
    scale = 1.0 / math.sqrt(head_dim)

    out = torch.empty_like(q)
    for b in range(bsz):
        for h in range(n_heads):
            kv_head_idx = h // kv_groups
            q_vec = q[b, h, 0, :].float()  # [head_dim]
            k_mat = k[b, kv_head_idx, :, :].float()  # [kv_len, head_dim]
            v_mat = v[b, kv_head_idx, :, :].float()  # [kv_len, head_dim]

            scores = (k_mat @ q_vec) * scale  # [kv_len]
            probs = torch.softmax(scores, dim=0)  # [kv_len]
            result = (probs.unsqueeze(1) * v_mat).sum(dim=0)  # [head_dim]
            out[b, h, 0, :] = result.to(q.dtype)
    return out


# ---------------------------------------------------------------------------
# Guard predicate tests (all run on CPU — no CUDA required)
# ---------------------------------------------------------------------------


def test_cannot_use_when_triton_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Predicate returns False when Triton is not installed."""
    import llm_cuda.kernels.triton.decode_attention as mod

    original = mod.triton
    try:
        monkeypatch.setattr(mod, "triton", None)
        q = torch.randn(1, 4, 1, 16)
        k = torch.randn(1, 2, 8, 16)
        v = torch.randn_like(k)
        assert can_use_triton_decode_attention(q, k, v) is False
    finally:
        mod.triton = original


def test_cannot_use_on_cpu() -> None:
    q = torch.randn(1, 4, 1, 16, dtype=torch.float16)
    k = torch.randn(1, 2, 8, 16, dtype=torch.float16)
    v = torch.randn_like(k)
    assert can_use_triton_decode_attention(q, k, v) is False


def test_cannot_use_with_attention_mask() -> None:
    q = torch.randn(1, 4, 1, 16, dtype=torch.float16)
    k = torch.randn(1, 2, 8, 16, dtype=torch.float16)
    v = torch.randn_like(k)
    mask = torch.zeros(1, 1, 1, 8, dtype=torch.float16)
    assert can_use_triton_decode_attention(q, k, v, attention_mask=mask) is False


def test_cannot_use_when_q_seq_len_ne_one() -> None:
    q = torch.randn(1, 4, 3, 16, dtype=torch.float16)
    k = torch.randn(1, 2, 8, 16, dtype=torch.float16)
    v = torch.randn_like(k)
    assert can_use_triton_decode_attention(q, k, v) is False


def test_cannot_use_when_head_dim_mismatch() -> None:
    q = torch.randn(1, 4, 1, 16, dtype=torch.float16)
    k = torch.randn(1, 2, 8, 32, dtype=torch.float16)
    v = torch.randn_like(k)
    assert can_use_triton_decode_attention(q, k, v) is False


def test_cannot_use_when_kv_heads_not_divisor() -> None:
    # 5 Q-heads, 3 KV-heads: 5 % 3 != 0
    q = torch.randn(1, 5, 1, 16, dtype=torch.float16)
    k = torch.randn(1, 3, 8, 16, dtype=torch.float16)
    v = torch.randn_like(k)
    assert can_use_triton_decode_attention(q, k, v) is False


def test_cannot_use_when_kv_len_zero() -> None:
    q = torch.randn(1, 4, 1, 16, dtype=torch.float16)
    k = torch.empty(1, 2, 0, 16, dtype=torch.float16)
    v = torch.empty_like(k)
    assert can_use_triton_decode_attention(q, k, v) is False


def test_cannot_use_when_head_dim_too_large() -> None:
    q = torch.randn(1, 4, 1, 512, dtype=torch.float16)
    k = torch.randn(1, 2, 8, 512, dtype=torch.float16)
    v = torch.randn_like(k)
    assert can_use_triton_decode_attention(q, k, v) is False


def test_cannot_use_float64_dtype() -> None:
    q = torch.randn(1, 4, 1, 16, dtype=torch.float64)
    k = torch.randn(1, 2, 8, 16, dtype=torch.float64)
    v = torch.randn_like(k)
    assert can_use_triton_decode_attention(q, k, v) is False


# ---------------------------------------------------------------------------
# Correctness tests (CPU fallback via reference; CUDA when available)
# ---------------------------------------------------------------------------


def _run_correctness(
    bsz: int,
    n_heads: int,
    n_kv_heads: int,
    kv_len: int,
    head_dim: int,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> None:
    torch.manual_seed(_TEST_SEED)
    q = torch.randn(bsz, n_heads, 1, head_dim, dtype=dtype, device=device)
    k = torch.randn(bsz, n_kv_heads, kv_len, head_dim, dtype=dtype, device=device)
    v = torch.randn(bsz, n_kv_heads, kv_len, head_dim, dtype=dtype, device=device)

    ref = _reference_decode_attention(q, k, v)

    if device == "cpu":
        # On CPU, the Triton kernel is not available; use the Python reference
        # to validate the reference itself is self-consistent (shape/dtype).
        assert ref.shape == q.shape
        assert ref.dtype == q.dtype
        return

    out = triton_decode_attention(q, k, v)
    assert out.shape == q.shape
    assert out.dtype == q.dtype
    assert torch.allclose(out.float(), ref.float(), atol=atol, rtol=rtol), (
        f"Max abs diff: {(out.float() - ref.float()).abs().max().item():.6f}"
    )


def test_reference_mha_cpu_shapes() -> None:
    """Reference produces correct shape/dtype on CPU (no CUDA required)."""
    _run_correctness(bsz=2, n_heads=4, n_kv_heads=4, kv_len=16, head_dim=16)


def test_reference_gqa_cpu_shapes() -> None:
    """Reference handles GQA shapes on CPU."""
    _run_correctness(bsz=1, n_heads=8, n_kv_heads=4, kv_len=32, head_dim=16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_decode_attention_mha_fp32() -> None:
    """MHA (n_heads == n_kv_heads) correctness in float32."""
    _run_correctness(
        bsz=2,
        n_heads=4,
        n_kv_heads=4,
        kv_len=64,
        head_dim=32,
        dtype=torch.float32,
        device="cuda",
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_decode_attention_gqa_fp32() -> None:
    """GQA (n_heads=8, n_kv_heads=4) correctness in float32."""
    _run_correctness(
        bsz=2,
        n_heads=8,
        n_kv_heads=4,
        kv_len=64,
        head_dim=32,
        dtype=torch.float32,
        device="cuda",
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_decode_attention_gqa_fp16() -> None:
    """GQA correctness with float16 dtype."""
    _run_correctness(
        bsz=1,
        n_heads=16,
        n_kv_heads=8,
        kv_len=128,
        head_dim=64,
        dtype=torch.float16,
        device="cuda",
        atol=1e-2,
        rtol=1e-2,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_decode_attention_gqa_bf16() -> None:
    """GQA correctness with bfloat16 dtype."""
    _run_correctness(
        bsz=1,
        n_heads=16,
        n_kv_heads=8,
        kv_len=128,
        head_dim=64,
        dtype=torch.bfloat16,
        device="cuda",
        atol=2e-2,
        rtol=2e-2,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_decode_attention_single_kv_position() -> None:
    """Edge case: kv_len == 1 (first decode step after a single-token prompt)."""
    _run_correctness(
        bsz=1,
        n_heads=4,
        n_kv_heads=2,
        kv_len=1,
        head_dim=16,
        dtype=torch.float32,
        device="cuda",
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_decode_attention_long_kv_cache() -> None:
    """Large KV cache forces multiple BLOCK_N tiles (tests tiling logic)."""
    _run_correctness(
        bsz=1,
        n_heads=8,
        n_kv_heads=4,
        kv_len=1024,
        head_dim=64,
        dtype=torch.float32,
        device="cuda",
        atol=1e-3,
        rtol=1e-3,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_decode_attention_head_dim_not_power_of_two() -> None:
    """head_dim=48 is not a power of 2 — BLOCK_D is padded to 64."""
    _run_correctness(
        bsz=1,
        n_heads=4,
        n_kv_heads=2,
        kv_len=32,
        head_dim=48,
        dtype=torch.float32,
        device="cuda",
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_decode_attention_batch_size_gt_one() -> None:
    """Batch size > 1 exercises the batch-stride logic in the kernel."""
    _run_correctness(
        bsz=4,
        n_heads=8,
        n_kv_heads=4,
        kv_len=64,
        head_dim=32,
        dtype=torch.float32,
        device="cuda",
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_can_use_predicate_true_on_cuda() -> None:
    """Guard predicate returns True for valid CUDA inputs."""
    q = torch.randn(1, 8, 1, 32, dtype=torch.float16, device="cuda")
    k = torch.randn(1, 4, 16, 32, dtype=torch.float16, device="cuda")
    v = torch.randn_like(k)
    assert can_use_triton_decode_attention(q, k, v) is True
