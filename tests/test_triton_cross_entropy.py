"""Tests for the Triton fused cross-entropy kernel.

Each test runs on CPU (PyTorch fallback) unconditionally and is skipped on CUDA when
the Triton kernel path is exercised.  Numerical tolerances match those used by the
existing attention/SwiGLU kernel tests in this repo.
"""

import pytest
import torch
import torch.nn.functional as F

from llm_cuda.kernels.triton.cross_entropy import (
    can_use_triton_cross_entropy,
    triton_cross_entropy,
)


def _make_logits(n: int, v: int, dtype=torch.float32, device="cpu", seed: int = 42):
    torch.manual_seed(seed)
    return torch.randn(n, v, dtype=dtype, device=device, requires_grad=True)


def _make_labels(n: int, v: int, ignore_frac: float = 0.0, seed: int = 7, device="cpu"):
    torch.manual_seed(seed)
    labels = torch.randint(0, v, (n,), device=device)
    if ignore_frac > 0.0:
        n_ignore = max(1, int(n * ignore_frac))
        labels[:n_ignore] = -100
    return labels


# ---------------------------------------------------------------------------
# Utility: forward parity with F.cross_entropy
# ---------------------------------------------------------------------------

class TestForwardParity:
    """Fused kernel output must match F.cross_entropy to within FP32 tolerance."""

    @pytest.mark.parametrize("n,v", [(8, 64), (32, 512), (64, 1024)])
    def test_mean_reduction_cpu(self, n, v):
        logits = _make_logits(n, v)
        labels = _make_labels(n, v)

        ref = F.cross_entropy(logits, labels, reduction="mean")
        got = triton_cross_entropy(logits, labels, reduction="mean")

        assert torch.allclose(ref, got, atol=1e-5, rtol=1e-5), (
            f"mean loss mismatch: ref={ref.item():.6f}, got={got.item():.6f}"
        )

    @pytest.mark.parametrize("n,v", [(8, 64), (16, 256)])
    def test_sum_reduction_cpu(self, n, v):
        logits = _make_logits(n, v)
        labels = _make_labels(n, v)

        ref = F.cross_entropy(logits, labels, reduction="sum")
        got = triton_cross_entropy(logits, labels, reduction="sum")

        assert torch.allclose(ref, got, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("n,v", [(8, 64)])
    def test_none_reduction_cpu(self, n, v):
        logits = _make_logits(n, v)
        labels = _make_labels(n, v)

        ref = F.cross_entropy(logits, labels, reduction="none")
        got = triton_cross_entropy(logits, labels, reduction="none")

        assert got.shape == ref.shape
        assert torch.allclose(ref, got.float(), atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("n,v,ignore_frac", [(16, 128, 0.25), (32, 256, 0.5)])
    def test_ignore_index_cpu(self, n, v, ignore_frac):
        logits = _make_logits(n, v)
        labels = _make_labels(n, v, ignore_frac=ignore_frac)

        ref = F.cross_entropy(logits, labels, ignore_index=-100, reduction="mean")
        got = triton_cross_entropy(logits, labels, ignore_index=-100, reduction="mean")

        assert torch.allclose(ref, got, atol=1e-5, rtol=1e-5), (
            f"ignore_index loss mismatch: ref={ref.item():.6f}, got={got.item():.6f}"
        )


# ---------------------------------------------------------------------------
# Gradient parity
# ---------------------------------------------------------------------------

class TestBackwardParity:
    """dlogits from the fused kernel must match torch autograd."""

    @pytest.mark.parametrize("n,v", [(8, 64), (16, 256)])
    def test_dlogits_cpu(self, n, v):
        torch.manual_seed(0)
        logits_ref = _make_logits(n, v)
        logits_fused = logits_ref.detach().clone().requires_grad_(True)
        labels = _make_labels(n, v)

        loss_ref = F.cross_entropy(logits_ref, labels, reduction="mean")
        loss_ref.backward()

        loss_fused = triton_cross_entropy(logits_fused, labels, reduction="mean")
        loss_fused.backward()

        assert logits_ref.grad is not None
        assert logits_fused.grad is not None
        assert torch.allclose(logits_ref.grad, logits_fused.grad, atol=1e-5, rtol=1e-5), (
            "gradient mismatch between F.cross_entropy and triton_cross_entropy"
        )

    @pytest.mark.parametrize("n,v", [(16, 128)])
    def test_dlogits_ignore_index_cpu(self, n, v):
        torch.manual_seed(1)
        logits_ref = _make_logits(n, v)
        logits_fused = logits_ref.detach().clone().requires_grad_(True)
        labels = _make_labels(n, v, ignore_frac=0.25)

        loss_ref = F.cross_entropy(logits_ref, labels, ignore_index=-100, reduction="mean")
        loss_ref.backward()

        loss_fused = triton_cross_entropy(logits_fused, labels, ignore_index=-100, reduction="mean")
        loss_fused.backward()

        assert torch.allclose(logits_ref.grad, logits_fused.grad, atol=1e-5, rtol=1e-5)

        # Ignored positions must have zero gradient.
        ignore_mask = labels == -100
        assert (logits_fused.grad[ignore_mask].abs() < 1e-9).all(), (
            "non-zero gradient found at ignored positions"
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_token(self):
        logits = _make_logits(1, 32)
        labels = _make_labels(1, 32)
        ref = F.cross_entropy(logits, labels)
        got = triton_cross_entropy(logits, labels)
        assert torch.allclose(ref, got, atol=1e-5)

    def test_all_ignored(self):
        """All positions ignored → loss should be 0 (Triton path) or nan (fallback)."""
        logits = _make_logits(8, 64)
        labels = torch.full((8,), -100)
        got = triton_cross_entropy(logits, labels, reduction="mean")
        # The Triton path uses n_valid.clamp(min=1) and returns 0.0; the CPU fallback
        # delegates to F.cross_entropy which returns nan (mean of an empty set).
        # Both are acceptable — the important property is that no gradient flows from
        # ignored positions.
        assert torch.isnan(got) or got.item() == 0.0

    def test_large_vocab_cpu(self):
        """Smoke test with a realistic 128K vocab on CPU (fallback path)."""
        n, v = 4, 128256
        logits = _make_logits(n, v)
        labels = _make_labels(n, v)
        ref = F.cross_entropy(logits, labels)
        got = triton_cross_entropy(logits, labels)
        assert torch.allclose(ref, got, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# CUDA-only tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCuda:
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    @pytest.mark.parametrize("n,v", [(32, 1024), (64, 4096)])
    def test_forward_parity_cuda(self, dtype, n, v):
        logits_cpu = _make_logits(n, v, dtype=torch.float32)
        labels = _make_labels(n, v)

        logits_cuda = logits_cpu.detach().cuda().to(dtype).requires_grad_(True)
        labels_cuda = labels.cuda()

        ref = F.cross_entropy(logits_cuda.float(), labels_cuda, reduction="mean")
        got = triton_cross_entropy(logits_cuda, labels_cuda, reduction="mean")

        assert can_use_triton_cross_entropy(logits_cuda, labels_cuda), (
            "Expected Triton kernel to be available on CUDA"
        )
        assert torch.allclose(ref.float(), got.float(), atol=1e-3, rtol=1e-3), (
            f"CUDA forward mismatch dtype={dtype}: ref={ref.item():.6f} got={got.item():.6f}"
        )

    @pytest.mark.parametrize("dtype", [torch.float32])
    def test_backward_parity_cuda(self, dtype):
        n, v = 16, 512
        logits_cpu = _make_logits(n, v)
        labels = _make_labels(n, v)

        logits_ref = logits_cpu.detach().cuda().to(dtype).requires_grad_(True)
        logits_fused = logits_cpu.detach().cuda().to(dtype).requires_grad_(True)
        labels_cuda = labels.cuda()

        F.cross_entropy(logits_ref, labels_cuda, reduction="mean").backward()
        triton_cross_entropy(logits_fused, labels_cuda, reduction="mean").backward()

        assert torch.allclose(logits_ref.grad.float(), logits_fused.grad.float(), atol=1e-4, rtol=1e-4)

    def test_large_vocab_cuda(self):
        """Fused kernel vs F.cross_entropy at Llama 3's vocab size."""
        n, v = 16, 128256
        logits = torch.randn(n, v, dtype=torch.float16, device="cuda")
        labels = torch.randint(0, v, (n,), device="cuda")

        ref = F.cross_entropy(logits.float(), labels, reduction="mean")
        got = triton_cross_entropy(logits, labels, reduction="mean")

        assert torch.allclose(ref.float(), got.float(), atol=1e-2, rtol=1e-2), (
            f"large-vocab mismatch: ref={ref.item():.4f} got={got.item():.4f}"
        )
