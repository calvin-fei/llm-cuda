import torch

from llm_cuda.kernels.cuda.attention import cuda_causal_attention_extension
from llm_cuda.kernels.cuda.rmsnorm import cuda_extension_status, cuda_rms_norm, cuda_rms_norm_extension
from llm_cuda.kernels.cuda.swiglu import cuda_swiglu_extension


def _reference_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    var = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(var + eps) * weight


def test_cuda_rms_norm_cpu_fallback_matches_reference() -> None:
    torch.manual_seed(40)
    x = torch.randn(2, 4, 16, dtype=torch.float32)
    weight = torch.randn(16, dtype=torch.float32)
    eps = 1e-5

    out = cuda_rms_norm(x, weight, eps)
    ref = _reference_rms_norm(x, weight, eps)

    assert out.shape == ref.shape
    assert torch.allclose(out, ref, atol=1e-6, rtol=1e-6)


def test_cuda_rms_norm_extension_returns_none_on_cpu() -> None:
    x = torch.randn(2, 4, 16, dtype=torch.float32)
    weight = torch.randn(16, dtype=torch.float32)

    out = cuda_rms_norm_extension(x, weight, 1e-5)
    assert out is None


def test_cuda_extension_status_reports_string() -> None:
    status = cuda_extension_status()
    assert isinstance(status, str)
    assert len(status) > 0


def test_cuda_swiglu_extension_returns_none_on_cpu() -> None:
    gate = torch.randn(2, 4, 16, dtype=torch.float32)
    up = torch.randn_like(gate)
    out = cuda_swiglu_extension(gate, up)
    assert out is None


def test_cuda_attention_extension_returns_none_on_cpu() -> None:
    q = torch.randn(1, 2, 8, 16, dtype=torch.float32)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    out = cuda_causal_attention_extension(q, k, v)
    assert out is None
