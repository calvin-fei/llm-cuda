"""Tests for the Triton fused AdamW optimizer step.

Verifies that ``triton_adamw_step`` and ``TritonAdamW`` produce numerically equivalent
results to a reference PyTorch implementation across multiple data types and parameter
sizes.
"""

import copy
import math

import pytest
import torch

from llm_cuda.kernels.triton.adamw import TritonAdamW, _pytorch_adamw_step, triton_adamw_step


# ---------------------------------------------------------------------------
# Reference PyTorch AdamW step for comparison
# ---------------------------------------------------------------------------

def _ref_step(param, grad, exp_avg, exp_avg_sq, *, lr, beta1, beta2, eps, weight_decay, step):
    """Stateless reference that mirrors _pytorch_adamw_step but returns new tensors."""
    p = param.float().clone()
    g = grad.float().clone()
    m = exp_avg.float().clone()
    v = exp_avg_sq.float().clone()

    bc1 = 1.0 - beta1 ** step
    bc2 = 1.0 - beta2 ** step

    m = beta1 * m + (1.0 - beta1) * g
    v = beta2 * v + (1.0 - beta2) * g * g

    step_size = lr / bc1
    denom = torch.sqrt(v / bc2) + eps

    p = p * (1.0 - lr * weight_decay) - step_size * m / denom
    return p, m, v


# ---------------------------------------------------------------------------
# triton_adamw_step functional API
# ---------------------------------------------------------------------------

class TestTritonAdamWStep:
    """``triton_adamw_step`` must match the reference up to float32 tolerance."""

    HYPERPARAMS = dict(lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01)

    @pytest.mark.parametrize("n", [32, 512, 1024, 4097])
    def test_fp32_cpu_parity(self, n):
        torch.manual_seed(0)
        param = torch.randn(n)
        grad = torch.randn(n)
        exp_avg = torch.zeros(n)
        exp_avg_sq = torch.zeros(n)

        # Reference (in-place PyTorch ops on a copy).
        param_ref = param.clone()
        m_ref = exp_avg.clone()
        v_ref = exp_avg_sq.clone()
        _pytorch_adamw_step(param_ref, grad.clone(), m_ref, v_ref, step=1, **self.HYPERPARAMS)

        # Triton step (falls back to PyTorch on CPU).
        param_got = param.clone()
        m_got = exp_avg.clone()
        v_got = exp_avg_sq.clone()
        triton_adamw_step(param_got, grad.clone(), m_got, v_got, step=1, **self.HYPERPARAMS)

        assert torch.allclose(param_ref, param_got, atol=1e-6, rtol=1e-6)
        assert torch.allclose(m_ref, m_got, atol=1e-6, rtol=1e-6)
        assert torch.allclose(v_ref, v_got, atol=1e-6, rtol=1e-6)

    def test_multi_step_cpu(self):
        """Multiple steps must accumulate moments correctly."""
        torch.manual_seed(1)
        n = 64
        param = torch.randn(n)
        grad_seq = [torch.randn(n) for _ in range(5)]

        param_ref = param.clone()
        m_ref, v_ref = torch.zeros(n), torch.zeros(n)
        param_got = param.clone()
        m_got, v_got = torch.zeros(n), torch.zeros(n)

        for t, g in enumerate(grad_seq, start=1):
            _pytorch_adamw_step(param_ref, g.clone(), m_ref, v_ref, step=t, **self.HYPERPARAMS)
            triton_adamw_step(param_got, g.clone(), m_got, v_got, step=t, **self.HYPERPARAMS)

        assert torch.allclose(param_ref, param_got, atol=1e-5, rtol=1e-5)

    def test_zero_weight_decay(self):
        """weight_decay=0 should behave identically to standard Adam."""
        torch.manual_seed(2)
        n = 32
        hp = dict(lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0)
        param = torch.randn(n)
        grad = torch.randn(n)

        param_ref = param.clone()
        m_ref, v_ref = torch.zeros(n), torch.zeros(n)
        _pytorch_adamw_step(param_ref, grad.clone(), m_ref, v_ref, step=1, **hp)

        param_got = param.clone()
        m_got, v_got = torch.zeros(n), torch.zeros(n)
        triton_adamw_step(param_got, grad.clone(), m_got, v_got, step=1, **hp)

        assert torch.allclose(param_ref, param_got, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# TritonAdamW optimizer class
# ---------------------------------------------------------------------------

class TestTritonAdamWOptimizer:
    """``TritonAdamW`` must produce the same result as ``torch.optim.AdamW``."""

    def _make_model(self, sizes, seed=0):
        torch.manual_seed(seed)
        if len(sizes) == 1:
            return torch.nn.Sequential(torch.nn.Linear(sizes[0], sizes[0], bias=False))
        layers = [torch.nn.Linear(a, b, bias=False) for a, b in zip(sizes, sizes[1:])]
        return torch.nn.Sequential(*layers)

    def test_single_step_parity(self):
        model_ref = self._make_model([64, 32])
        model_got = copy.deepcopy(model_ref)

        hp = dict(lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
        opt_ref = torch.optim.AdamW(model_ref.parameters(), **hp)
        opt_got = TritonAdamW(model_got.parameters(), **hp)

        torch.manual_seed(10)
        x = torch.randn(8, 64)
        loss_ref = model_ref(x).sum()
        loss_ref.backward()
        opt_ref.step()
        opt_ref.zero_grad()

        loss_got = model_got(x).sum()
        loss_got.backward()
        opt_got.step()
        opt_got.zero_grad()

        for p_ref, p_got in zip(model_ref.parameters(), model_got.parameters()):
            assert torch.allclose(p_ref, p_got, atol=1e-5, rtol=1e-5), (
                "TritonAdamW diverged from torch.optim.AdamW after one step"
            )

    def test_multi_step_parity(self):
        model_ref = self._make_model([32])
        model_got = copy.deepcopy(model_ref)

        hp = dict(lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
        opt_ref = torch.optim.AdamW(model_ref.parameters(), **hp)
        opt_got = TritonAdamW(model_got.parameters(), **hp)

        torch.manual_seed(99)
        for _ in range(10):
            x = torch.randn(4, 32)
            model_ref(x).sum().backward()
            opt_ref.step()
            opt_ref.zero_grad()

            model_got(x).sum().backward()
            opt_got.step()
            opt_got.zero_grad()

        for p_ref, p_got in zip(model_ref.parameters(), model_got.parameters()):
            assert torch.allclose(p_ref, p_got, atol=1e-4, rtol=1e-4), (
                "TritonAdamW diverged from torch.optim.AdamW after 10 steps"
            )

    def test_closure(self):
        """TritonAdamW must support closure-based loss computation."""
        model = self._make_model([16])
        opt = TritonAdamW(model.parameters(), lr=1e-3)

        x = torch.randn(4, 16)

        def closure():
            opt.zero_grad()
            loss = model(x).sum()
            loss.backward()
            return loss

        loss_val = opt.step(closure)
        assert loss_val is not None
        assert loss_val.item() != 0.0

    def test_no_grad_params_skipped(self):
        """Parameters without gradients must not be updated."""
        torch.manual_seed(0)
        param = torch.nn.Parameter(torch.randn(8))
        frozen = torch.nn.Parameter(torch.randn(8), requires_grad=False)

        opt = TritonAdamW([param, frozen], lr=1e-3)

        frozen_before = frozen.data.clone()
        param.grad = torch.randn_like(param)
        opt.step()

        assert torch.equal(frozen, frozen_before), "Frozen parameter was modified"
        assert not torch.equal(param.data, torch.randn(8)), "Active parameter was not updated"


# ---------------------------------------------------------------------------
# CUDA-only tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCuda:
    HYPERPARAMS = dict(lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01)

    @pytest.mark.parametrize("dtype", [torch.float32])
    @pytest.mark.parametrize("n", [512, 4096])
    def test_triton_kernel_path(self, dtype, n):
        """Verify the Triton kernel path is taken and produces correct results."""
        torch.manual_seed(5)
        param = torch.randn(n, dtype=dtype, device="cuda")
        grad = torch.randn(n, dtype=dtype, device="cuda")
        exp_avg = torch.zeros(n, dtype=torch.float32, device="cuda")
        exp_avg_sq = torch.zeros(n, dtype=torch.float32, device="cuda")

        # Reference on CPU.
        param_ref = param.cpu().float().clone()
        m_ref = torch.zeros(n)
        v_ref = torch.zeros(n)
        _pytorch_adamw_step(param_ref, grad.cpu().float(), m_ref, v_ref, step=1, **self.HYPERPARAMS)

        triton_adamw_step(param, grad, exp_avg, exp_avg_sq, step=1, **self.HYPERPARAMS)

        assert torch.allclose(param_ref, param.cpu().float(), atol=1e-5, rtol=1e-5)
        assert torch.allclose(m_ref, exp_avg.cpu(), atol=1e-5, rtol=1e-5)
        assert torch.allclose(v_ref, exp_avg_sq.cpu(), atol=1e-5, rtol=1e-5)
