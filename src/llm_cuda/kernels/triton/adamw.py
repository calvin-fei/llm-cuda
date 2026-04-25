"""Triton fused AdamW optimizer step kernel.

PyTorch's ``torch.optim.AdamW`` issues one CUDA kernel per arithmetic operation:
weight decay, first-moment update, second-moment update, bias correction (×2), and the
final parameter update — roughly **6 kernel launches and 6 full-parameter reads/writes**
per step.  For a model with hundreds of millions of parameters spread across thousands
of tensors, this serialises GPU work and saturates the PCIe/NVLink bus.

This module provides a **single-kernel fused AdamW step** implemented in Triton.  The
full update is computed in a single pass over each parameter block:

.. code-block:: none

    m_t  = β₁·m_{t-1} + (1 − β₁)·g
    v_t  = β₂·v_{t-1} + (1 − β₂)·g²
    p_t  = p_{t-1} · (1 − lr·λ)  −  lr · (m_t / bc₁) / (√(v_t / bc₂) + ε)

where ``bc₁ = 1 − β₁ᵗ`` and ``bc₂ = 1 − β₂ᵗ`` are the bias-correction scalars.

Key properties
--------------
* **Single pass**: one kernel load/store per parameter element instead of six.
* **Float-32 arithmetic**: all moment updates are accumulated in FP32; parameter writes
  are cast back to the original dtype, enabling both FP32 and BF16/FP16 model weights.
* **Drop-in ``Optimizer``**: ``TritonAdamW`` follows the ``torch.optim.Optimizer`` API
  and can replace ``torch.optim.AdamW`` without any model-side changes.

Public API
----------
``triton_adamw_step(param, grad, exp_avg, exp_avg_sq, lr, beta1, beta2, eps, weight_decay, step)``
    Functional interface — updates ``param``, ``exp_avg``, ``exp_avg_sq`` in-place.

``TritonAdamW``
    ``torch.optim.Optimizer`` subclass with the same signature as
    ``torch.optim.AdamW``.
"""

import math

import torch
from torch.optim import Optimizer

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None

# Elements processed per Triton program.  512 matches 4 cache lines for FP32
# (512 × 4 B = 2 KB) and maps comfortably onto 4 warps × 32 threads × 4 elements.
_BLOCK_SIZE: int = 512


if triton is not None:

    @triton.jit
    def _adamw_step_kernel(
        param_ptr,
        grad_ptr,
        exp_avg_ptr,
        exp_avg_sq_ptr,
        n_elements,
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
        bias_correction1,
        bias_correction2,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Fused AdamW update — one Triton program per ``BLOCK_SIZE`` parameter elements.

        All arithmetic is performed in FP32.  ``param`` is read, updated, and written
        back in its original dtype (FP16, BF16, or FP32).  ``exp_avg`` and
        ``exp_avg_sq`` are always stored in FP32.

        The per-element update rule:

        .. code-block:: none

            m  ← β₁·m  + (1−β₁)·g
            v  ← β₂·v  + (1−β₂)·g²
            p  ← p·(1−lr·λ) − lr·(m/bc₁) / (√(v/bc₂) + ε)
        """
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements

        # Load all tensors and upcast to FP32 for arithmetic.
        p = tl.load(param_ptr + offs, mask=mask).to(tl.float32)
        g = tl.load(grad_ptr + offs, mask=mask).to(tl.float32)
        m = tl.load(exp_avg_ptr + offs, mask=mask).to(tl.float32)
        v = tl.load(exp_avg_sq_ptr + offs, mask=mask).to(tl.float32)

        # Update biased first and second moment estimates.
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * g * g

        # Bias-corrected step size.
        step_size = lr / bias_correction1
        denom = tl.sqrt(v / bias_correction2) + eps

        # Decoupled weight decay + parameter update (AdamW formulation).
        p = p * (1.0 - lr * weight_decay) - step_size * m / denom

        # Store moments in FP32.
        tl.store(exp_avg_ptr + offs, m, mask=mask)
        tl.store(exp_avg_sq_ptr + offs, v, mask=mask)

        # Store parameter in original dtype.
        param_raw = tl.load(param_ptr + offs, mask=mask)
        tl.store(param_ptr + offs, p.to(param_raw.dtype), mask=mask)


def _can_use_triton_adamw(param: torch.Tensor, grad: torch.Tensor) -> bool:
    if triton is None:
        return False
    if not (param.is_cuda and grad.is_cuda):
        return False
    if param.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        return False
    return True


def triton_adamw_step(
    param: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    *,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    step: int,
) -> None:
    """Perform a single fused AdamW step in-place.

    Updates ``param``, ``exp_avg``, and ``exp_avg_sq`` in-place using the Triton
    kernel when available; falls back to standard PyTorch ops otherwise.

    Args:
        param:        Parameter tensor (modified in-place).
        grad:         Gradient tensor (same shape and device as ``param``).
        exp_avg:      First-moment buffer (FP32, same shape as ``param``).
        exp_avg_sq:   Second-moment buffer (FP32, same shape as ``param``).
        lr:           Learning rate.
        beta1:        Exponential decay rate for the first moment.
        beta2:        Exponential decay rate for the second moment.
        eps:          Denominator epsilon for numerical stability.
        weight_decay: Decoupled weight decay coefficient.
        step:         Current optimiser step count (1-based).
    """
    if not _can_use_triton_adamw(param, grad):
        _pytorch_adamw_step(param, grad, exp_avg, exp_avg_sq, lr=lr, beta1=beta1,
                            beta2=beta2, eps=eps, weight_decay=weight_decay, step=step)
        return

    bias_correction1 = 1.0 - beta1 ** step
    bias_correction2 = 1.0 - beta2 ** step

    n_elements = param.numel()
    param_flat = param.view(-1)
    grad_flat = grad.contiguous().view(-1)
    exp_avg_flat = exp_avg.view(-1)
    exp_avg_sq_flat = exp_avg_sq.view(-1)

    block_size = _BLOCK_SIZE
    grid = (math.ceil(n_elements / block_size),)

    _adamw_step_kernel[grid](
        param_flat,
        grad_flat,
        exp_avg_flat,
        exp_avg_sq_flat,
        n_elements,
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
        bias_correction1,
        bias_correction2,
        BLOCK_SIZE=block_size,
    )


def _pytorch_adamw_step(
    param: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    *,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    step: int,
) -> None:
    """PyTorch reference implementation used as a fallback (CPU or unsupported dtype)."""
    bias_correction1 = 1.0 - beta1 ** step
    bias_correction2 = 1.0 - beta2 ** step

    # In-place moment updates.
    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

    step_size = lr / bias_correction1
    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

    # Decoupled weight decay.
    param.mul_(1.0 - lr * weight_decay)
    param.addcdiv_(exp_avg, denom, value=-step_size)


class TritonAdamW(Optimizer):
    """AdamW optimizer backed by a fused Triton kernel.

    Follows the ``torch.optim.AdamW`` interface exactly and can be used as a
    drop-in replacement.  On CUDA devices the Triton kernel fuses the six
    per-parameter operations into a single kernel launch; on CPU it falls back to
    the standard PyTorch implementation.

    Args:
        params:       Iterable of parameters or parameter groups.
        lr:           Learning rate (default: ``1e-3``).
        betas:        Exponential decay rates for moments (default: ``(0.9, 0.999)``).
        eps:          Denominator epsilon (default: ``1e-8``).
        weight_decay: Decoupled weight decay coefficient (default: ``1e-2``).

    Example::

        optimizer = TritonAdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
        for batch in dataloader:
            loss = compute_loss(model, batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("TritonAdamW does not support sparse gradients")

                state = self.state[p]

                # Lazy state initialisation on first step.
                if len(state) == 0:
                    state["step"] = 0
                    # Moments are always kept in FP32 regardless of param dtype.
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)

                state["step"] += 1

                triton_adamw_step(
                    p,
                    grad,
                    state["exp_avg"],
                    state["exp_avg_sq"],
                    lr=lr,
                    beta1=beta1,
                    beta2=beta2,
                    eps=eps,
                    weight_decay=weight_decay,
                    step=state["step"],
                )

        return loss
