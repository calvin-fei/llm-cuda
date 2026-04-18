import math

import pytest
import torch

from llm_cuda.kernels.triton.attention import (
    _select_attention_block_sizes,
    _should_use_cuda_sdpa,
    can_use_triton_fused_attention,
    triton_fused_causal_attention,
)


def _reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    seq_len = q.shape[-2]
    causal_mask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), dtype=q.dtype, device=q.device),
        diagonal=1,
    )
    scores = scores + causal_mask

    if attention_mask is not None:
        scores = scores + attention_mask

    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


def test_triton_attention_cpu_fallback_parity() -> None:
    torch.manual_seed(0)
    q = torch.randn(2, 4, 8, 16, dtype=torch.float32)
    k = torch.randn(2, 4, 8, 16, dtype=torch.float32)
    v = torch.randn(2, 4, 8, 16, dtype=torch.float32)

    out = triton_fused_causal_attention(q, k, v)
    ref = _reference_attention(q, k, v)

    assert out.shape == ref.shape
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_attention_tuning_block_sizes() -> None:
    bn_small, bd_small = _select_attention_block_sizes(seq_len=128, head_dim=64)
    bn_mid, bd_mid = _select_attention_block_sizes(seq_len=512, head_dim=64)
    bn_large, bd_large = _select_attention_block_sizes(seq_len=2048, head_dim=64)

    assert bn_small == 32
    assert bn_mid == 64
    assert bn_large == 128
    assert bd_small == 64
    assert bd_mid == 64
    assert bd_large == 64


def test_can_use_triton_attention_predicate_cpu_false() -> None:
    q = torch.randn(1, 2, 8, 32, dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    assert can_use_triton_fused_attention(q, k, v) is False


def test_should_use_sdpa_selector_cpu_false() -> None:
    q = torch.randn(1, 2, 1024, 64, dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    assert _should_use_cuda_sdpa(q, k, v, attention_mask=None) is False


def test_triton_attention_cpu_fallback_with_mask() -> None:
    torch.manual_seed(1)
    q = torch.randn(1, 2, 6, 8, dtype=torch.float32)
    k = torch.randn(1, 2, 6, 8, dtype=torch.float32)
    v = torch.randn(1, 2, 6, 8, dtype=torch.float32)
    mask = torch.zeros((1, 1, 1, 6), dtype=torch.float32)
    mask[..., -2:] = -1e4

    out = triton_fused_causal_attention(q, k, v, attention_mask=mask)
    ref = _reference_attention(q, k, v, attention_mask=mask)

    assert out.shape == ref.shape
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_triton_attention_cpu_fallback_backward_parity() -> None:
    torch.manual_seed(7)
    q_base = torch.randn(2, 3, 8, 16, dtype=torch.float32)
    k_base = torch.randn(2, 3, 8, 16, dtype=torch.float32)
    v_base = torch.randn(2, 3, 8, 16, dtype=torch.float32)
    grad_out = torch.randn(2, 3, 8, 16, dtype=torch.float32)

    q = q_base.clone().detach().requires_grad_(True)
    k = k_base.clone().detach().requires_grad_(True)
    v = v_base.clone().detach().requires_grad_(True)
    out = triton_fused_causal_attention(q, k, v)
    (out * grad_out).sum().backward()

    q_ref = q_base.clone().detach().requires_grad_(True)
    k_ref = k_base.clone().detach().requires_grad_(True)
    v_ref = v_base.clone().detach().requires_grad_(True)
    out_ref = _reference_attention(q_ref, k_ref, v_ref)
    (out_ref * grad_out).sum().backward()

    assert q.grad is not None and q_ref.grad is not None
    assert k.grad is not None and k_ref.grad is not None
    assert v.grad is not None and v_ref.grad is not None
    assert torch.allclose(q.grad, q_ref.grad, atol=1e-5, rtol=1e-5)
    assert torch.allclose(k.grad, k_ref.grad, atol=1e-5, rtol=1e-5)
    assert torch.allclose(v.grad, v_ref.grad, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton kernel parity test")
def test_triton_attention_cuda_backward_parity() -> None:
    torch.manual_seed(11)
    shape = (1, 2, 16, 32)
    dtype = torch.float16

    try:
        q_base = torch.randn(*shape, device="cuda", dtype=dtype)
        k_base = torch.randn(*shape, device="cuda", dtype=dtype)
        v_base = torch.randn(*shape, device="cuda", dtype=dtype)
        grad_out = torch.randn(*shape, device="cuda", dtype=dtype)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            pytest.skip("Skipping CUDA parity test due to CUDA OOM")
        raise

    try:
        q = q_base.clone().detach().requires_grad_(True)
        k = k_base.clone().detach().requires_grad_(True)
        v = v_base.clone().detach().requires_grad_(True)
        out = triton_fused_causal_attention(q, k, v)
        (out * grad_out).sum().backward()

        q_ref = q_base.clone().detach().requires_grad_(True)
        k_ref = k_base.clone().detach().requires_grad_(True)
        v_ref = v_base.clone().detach().requires_grad_(True)
        out_ref = _reference_attention(q_ref, k_ref, v_ref)
        (out_ref * grad_out).sum().backward()
    except Exception as exc:  # pragma: no cover - environment-dependent
        if "out of memory" in str(exc).lower():
            torch.cuda.empty_cache()
            pytest.skip("Skipping CUDA parity test due to CUDA OOM during execution")
        raise

    assert torch.allclose(out, out_ref, atol=3e-2, rtol=5e-2)
    assert q.grad is not None and q_ref.grad is not None
    assert k.grad is not None and k_ref.grad is not None
    assert v.grad is not None and v_ref.grad is not None
    assert torch.allclose(q.grad, q_ref.grad, atol=3e-2, rtol=6e-2)
    assert torch.allclose(k.grad, k_ref.grad, atol=3e-2, rtol=6e-2)
    assert torch.allclose(v.grad, v_ref.grad, atol=3e-2, rtol=6e-2)
