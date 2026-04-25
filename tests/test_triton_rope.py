"""Tests for the fused Triton RoPE kernel.

All correctness tests compare the kernel output against the original
PyTorch reference (rotate_half + element-wise ops).
"""

import math

import pytest
import torch

from llm_cuda.kernels.triton.rope import can_use_triton_rope, triton_apply_rope
from llm_cuda.models.llama3.rotary import (
    apply_rotary,
    build_inv_freq,
    build_rope_cache,
    rotate_half,
)


_TEST_SEED: int = 99


def _reference_apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Pure-PyTorch RoPE application: x * cos + rotate_half(x) * sin.

    Args:
        x:   ``[bsz, n_heads, seq, head_dim]``
        cos: ``[seq, head_dim]``
        sin: ``[seq, head_dim]``
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, head_dim]
    sin = sin.unsqueeze(0).unsqueeze(0)
    return (x * cos) + (rotate_half(x) * sin)


# ---------------------------------------------------------------------------
# Guard predicate tests (CPU, no CUDA required)
# ---------------------------------------------------------------------------


def test_cannot_use_when_triton_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    import llm_cuda.kernels.triton.rope as mod

    original = mod.triton
    try:
        monkeypatch.setattr(mod, "triton", None)
        x = torch.randn(1, 4, 4, 16)
        cos = torch.randn(4, 16)
        sin = torch.randn(4, 16)
        assert can_use_triton_rope(x, cos, sin) is False
    finally:
        mod.triton = original


def test_cannot_use_on_cpu() -> None:
    x = torch.randn(1, 4, 4, 16, dtype=torch.float32)
    cos = torch.randn(4, 16, dtype=torch.float32)
    sin = torch.randn(4, 16, dtype=torch.float32)
    assert can_use_triton_rope(x, cos, sin) is False


def test_cannot_use_with_odd_head_dim() -> None:
    # head_dim must be even for the half-split to work
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    x = torch.randn(1, 4, 4, 15, device="cuda", dtype=torch.float16)
    cos = torch.randn(4, 15, device="cuda", dtype=torch.float16)
    sin = torch.randn(4, 15, device="cuda", dtype=torch.float16)
    assert can_use_triton_rope(x, cos, sin) is False


def test_cannot_use_with_wrong_cos_shape() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    x = torch.randn(1, 4, 4, 16, device="cuda", dtype=torch.float16)
    # cos has wrong seq_len
    cos = torch.randn(3, 16, device="cuda", dtype=torch.float16)
    sin = torch.randn(4, 16, device="cuda", dtype=torch.float16)
    assert can_use_triton_rope(x, cos, sin) is False


def test_cannot_use_with_float64() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    x = torch.randn(1, 4, 4, 16, device="cuda", dtype=torch.float64)
    cos = torch.randn(4, 16, device="cuda", dtype=torch.float64)
    sin = torch.randn(4, 16, device="cuda", dtype=torch.float64)
    assert can_use_triton_rope(x, cos, sin) is False


# ---------------------------------------------------------------------------
# Correctness tests (CUDA required)
# ---------------------------------------------------------------------------


def _run_rope_correctness(
    bsz: int,
    n_heads: int,
    seq_len: int,
    head_dim: int,
    dtype: torch.dtype = torch.float32,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> None:
    torch.manual_seed(_TEST_SEED)
    x = torch.randn(bsz, n_heads, seq_len, head_dim, device="cuda", dtype=dtype)
    cos = torch.randn(seq_len, head_dim, device="cuda", dtype=dtype)
    sin = torch.randn(seq_len, head_dim, device="cuda", dtype=dtype)

    ref = _reference_apply_rope(x, cos, sin)
    out = triton_apply_rope(x, cos, sin)

    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert torch.allclose(out.float(), ref.float(), atol=atol, rtol=rtol), (
        f"Max abs diff: {(out.float() - ref.float()).abs().max().item():.6f}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_rope_fp32_basic() -> None:
    _run_rope_correctness(bsz=2, n_heads=4, seq_len=16, head_dim=32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_rope_fp16() -> None:
    _run_rope_correctness(
        bsz=1,
        n_heads=8,
        seq_len=32,
        head_dim=64,
        dtype=torch.float16,
        atol=1e-2,
        rtol=1e-2,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_rope_bf16() -> None:
    _run_rope_correctness(
        bsz=1,
        n_heads=8,
        seq_len=32,
        head_dim=64,
        dtype=torch.bfloat16,
        atol=2e-2,
        rtol=2e-2,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_rope_decode_step_seq1() -> None:
    """Decode step: seq_len=1 (most common hot path)."""
    _run_rope_correctness(
        bsz=1,
        n_heads=16,
        seq_len=1,
        head_dim=64,
        dtype=torch.float16,
        atol=1e-2,
        rtol=1e-2,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_rope_large_batch_heads() -> None:
    _run_rope_correctness(bsz=4, n_heads=16, seq_len=64, head_dim=64)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_rope_head_dim_128() -> None:
    _run_rope_correctness(bsz=1, n_heads=8, seq_len=16, head_dim=128)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_can_use_predicate_true_on_cuda() -> None:
    x = torch.randn(1, 8, 4, 64, device="cuda", dtype=torch.float16)
    cos = torch.randn(4, 64, device="cuda", dtype=torch.float16)
    sin = torch.randn(4, 64, device="cuda", dtype=torch.float16)
    assert can_use_triton_rope(x, cos, sin) is True


# ---------------------------------------------------------------------------
# build_inv_freq caching — no CUDA required
# ---------------------------------------------------------------------------


def test_build_inv_freq_shape() -> None:
    inv = build_inv_freq(head_dim=64, base=10000.0, device=torch.device("cpu"))
    assert inv.shape == (32,)
    assert inv.dtype == torch.float32


def test_build_inv_freq_values() -> None:
    """Verify inv_freq matches the values embedded in build_rope_cache."""
    head_dim = 32
    base = 500000.0
    inv = build_inv_freq(head_dim=head_dim, base=base, device=torch.device("cpu"))
    cos_cached, sin_cached = build_rope_cache(
        seq_len=4, head_dim=head_dim, base=base, device=torch.device("cpu"),
        dtype=torch.float32, inv_freq=inv,
    )
    cos_fresh, sin_fresh = build_rope_cache(
        seq_len=4, head_dim=head_dim, base=base, device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert torch.allclose(cos_cached, cos_fresh)
    assert torch.allclose(sin_cached, sin_fresh)


# ---------------------------------------------------------------------------
# apply_rotary dispatches to Triton on CUDA
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_apply_rotary_uses_triton_on_cuda() -> None:
    """apply_rotary produces correct output on CUDA (Triton kernel active)."""
    torch.manual_seed(_TEST_SEED)
    bsz, n_heads, n_kv_heads, seq_len, head_dim = 1, 8, 4, 16, 64
    dtype = torch.float16

    q = torch.randn(bsz, n_heads, seq_len, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(bsz, n_kv_heads, seq_len, head_dim, device="cuda", dtype=dtype)
    cos = torch.randn(seq_len, head_dim, device="cuda", dtype=dtype)
    sin = torch.randn(seq_len, head_dim, device="cuda", dtype=dtype)

    q_out, k_out = apply_rotary(q, k, cos, sin)

    q_ref = _reference_apply_rope(q, cos, sin)
    k_ref = _reference_apply_rope(k, cos, sin)

    assert torch.allclose(q_out.float(), q_ref.float(), atol=1e-2, rtol=1e-2)
    assert torch.allclose(k_out.float(), k_ref.float(), atol=1e-2, rtol=1e-2)
