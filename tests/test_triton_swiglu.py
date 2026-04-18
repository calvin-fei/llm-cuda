import pytest
import torch

from llm_cuda.kernels.triton.swiglu import triton_swiglu


def _reference_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.silu(gate) * up


def test_triton_swiglu_cpu_forward_parity() -> None:
    torch.manual_seed(21)
    gate = torch.randn(2, 4, 16, dtype=torch.float32)
    up = torch.randn_like(gate)

    out = triton_swiglu(gate, up)
    ref = _reference_swiglu(gate, up)

    assert out.shape == ref.shape
    assert torch.allclose(out, ref, atol=1e-6, rtol=1e-6)


def test_triton_swiglu_cpu_backward_parity() -> None:
    torch.manual_seed(22)
    gate_base = torch.randn(2, 3, 16, dtype=torch.float32)
    up_base = torch.randn_like(gate_base)
    grad_out = torch.randn_like(gate_base)

    gate = gate_base.clone().detach().requires_grad_(True)
    up = up_base.clone().detach().requires_grad_(True)
    out = triton_swiglu(gate, up)
    (out * grad_out).sum().backward()

    gate_ref = gate_base.clone().detach().requires_grad_(True)
    up_ref = up_base.clone().detach().requires_grad_(True)
    out_ref = _reference_swiglu(gate_ref, up_ref)
    (out_ref * grad_out).sum().backward()

    assert gate.grad is not None and gate_ref.grad is not None
    assert up.grad is not None and up_ref.grad is not None
    assert torch.allclose(gate.grad, gate_ref.grad, atol=1e-6, rtol=1e-6)
    assert torch.allclose(up.grad, up_ref.grad, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton SwiGLU parity test")
def test_triton_swiglu_cuda_forward_backward_parity() -> None:
    torch.manual_seed(23)
    shape = (1, 8, 64)
    dtype = torch.float16

    try:
        gate_base = torch.randn(*shape, device="cuda", dtype=dtype)
        up_base = torch.randn_like(gate_base)
        grad_out = torch.randn_like(gate_base)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            pytest.skip("Skipping CUDA SwiGLU parity test due to CUDA OOM")
        raise

    try:
        gate = gate_base.clone().detach().requires_grad_(True)
        up = up_base.clone().detach().requires_grad_(True)
        out = triton_swiglu(gate, up)
        (out * grad_out).sum().backward()

        gate_ref = gate_base.clone().detach().requires_grad_(True)
        up_ref = up_base.clone().detach().requires_grad_(True)
        out_ref = _reference_swiglu(gate_ref, up_ref)
        (out_ref * grad_out).sum().backward()
    except Exception as exc:  # pragma: no cover
        if "out of memory" in str(exc).lower():
            torch.cuda.empty_cache()
            pytest.skip("Skipping CUDA SwiGLU parity test due to CUDA OOM during execution")
        raise

    assert torch.allclose(out, out_ref, atol=3e-2, rtol=5e-2)
    assert gate.grad is not None and gate_ref.grad is not None
    assert up.grad is not None and up_ref.grad is not None
    assert torch.allclose(gate.grad, gate_ref.grad, atol=3e-2, rtol=5e-2)
    assert torch.allclose(up.grad, up_ref.grad, atol=3e-2, rtol=5e-2)
