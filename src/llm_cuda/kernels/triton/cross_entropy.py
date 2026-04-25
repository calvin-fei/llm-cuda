"""Triton fused cross-entropy loss kernel for large-vocabulary LLMs.

The standard approach—``F.cross_entropy(logits.view(-1, V), labels.view(-1))``—first
materialises the full ``[N, V]`` softmax probability matrix and then reduces it.  For
Llama 3 with V = 128 256, a single prefill step at batch 4 × sequence 2 048 creates a
~8 GB FP16 intermediate, making large-batch training GPU-memory bound at the loss step
alone.

This module replaces that with a **two-pass tiled Triton kernel** that processes the
vocabulary in ``BLOCK_V``-wide tiles, never materialising the full softmax matrix:

* **Forward pass**: finds the row-wise max, computes ``log-sum-exp``, and returns
  per-token loss in ``O(N)`` extra memory (just the ``lse`` vector).
* **Backward pass**: recomputes ``softmax(logits[i])`` tile-by-tile from the saved
  ``lse[i]``, subtracts the one-hot target, and writes ``dlogits`` — again without any
  ``[N, V]`` intermediate.

Peak memory drops from ``O(N × V)`` to ``O(N)`` relative to the PyTorch baseline.

Public API
----------
``triton_cross_entropy(logits, labels, ignore_index=-100, reduction="mean")``
    Drop-in replacement for ``F.cross_entropy``.  Falls back to the PyTorch
    implementation when Triton is unavailable or the inputs are on CPU.
"""

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None

# Vocabulary tile width.  1024 elements × 2 bytes (FP16) = 2 KB per tile, which fits
# comfortably in L1.  Large enough to amortise loop overhead across the 128K vocab.
_BLOCK_V: int = 1024


if triton is not None:

    @triton.jit
    def _ce_fwd_kernel(
        logits_ptr,
        labels_ptr,
        loss_ptr,
        lse_ptr,
        stride_logits_row,
        n_cols,
        ignore_index,
        BLOCK_V: tl.constexpr,
    ):
        """Compute per-token cross-entropy loss in two tiled passes over the vocab.

        Grid: ``(N,)`` where ``N = B * seq_len`` (one program per token position).

        Pass 1 — find ``row_max``:
            Iterates over ``[0, n_cols)`` in ``BLOCK_V`` tiles; accumulates the
            element-wise maximum without storing any intermediate tensor.

        Pass 2 — compute ``log-sum-exp``:
            Re-reads the same tiles, subtracts ``row_max`` for numerical stability,
            sums ``exp(x - row_max)``, then finalises ``lse = log(sum) + row_max``.

        The per-token loss is ``lse - logits[row, label]`` (or 0 for ignored positions).
        Both ``loss`` and ``lse`` are stored in FP32.
        """
        row = tl.program_id(0)
        label = tl.load(labels_ptr + row)
        is_valid = label != ignore_index

        # ------------------------------------------------------------------
        # Pass 1: row-wise maximum for numerical stability.
        # ------------------------------------------------------------------
        row_max = -float("inf")
        for start in tl.range(0, n_cols, BLOCK_V):
            offs = start + tl.arange(0, BLOCK_V)
            mask = offs < n_cols
            x = tl.load(
                logits_ptr + row * stride_logits_row + offs,
                mask=mask,
                other=-float("inf"),
            ).to(tl.float32)
            row_max = tl.maximum(row_max, tl.max(x, axis=0))

        # ------------------------------------------------------------------
        # Pass 2: log-sum-exp accumulation.
        # ------------------------------------------------------------------
        lse_sum = 0.0
        for start in tl.range(0, n_cols, BLOCK_V):
            offs = start + tl.arange(0, BLOCK_V)
            mask = offs < n_cols
            x = tl.load(
                logits_ptr + row * stride_logits_row + offs,
                mask=mask,
                other=-float("inf"),
            ).to(tl.float32)
            lse_sum += tl.sum(tl.exp(x - row_max), axis=0)

        lse = tl.log(lse_sum) + row_max
        tl.store(lse_ptr + row, lse)

        # ------------------------------------------------------------------
        # Per-token loss: lse − logits[label].
        # ------------------------------------------------------------------
        target_logit = tl.load(
            logits_ptr + row * stride_logits_row + label,
            mask=is_valid,
            other=0.0,
        ).to(tl.float32)
        loss_val = tl.where(is_valid, lse - target_logit, 0.0)
        tl.store(loss_ptr + row, loss_val)

    @triton.jit
    def _ce_bwd_kernel(
        logits_ptr,
        labels_ptr,
        lse_ptr,
        grad_loss_ptr,
        dlogits_ptr,
        stride_logits_row,
        stride_dlogits_row,
        n_cols,
        ignore_index,
        BLOCK_V: tl.constexpr,
    ):
        """Compute ``dlogits = grad_loss * (softmax(logits) - one_hot(label))``.

        Grid: ``(N,)`` — one program per token position.

        Each iteration of the vocab loop:
        1. Loads the logit tile.
        2. Recomputes ``softmax`` values from the saved ``lse`` (no re-materialised
           softmax matrix — only one tile at a time is live in registers).
        3. Subtracts the one-hot contribution at the target position.
        4. Multiplies by ``grad_loss`` and writes ``dlogits``.

        For ignored positions (``label == ignore_index``) the entire gradient row is
        zero, which is enforced via ``tl.where``.
        """
        row = tl.program_id(0)
        label = tl.load(labels_ptr + row)
        lse = tl.load(lse_ptr + row)
        grad = tl.load(grad_loss_ptr + row).to(tl.float32)
        is_valid = label != ignore_index

        for start in tl.range(0, n_cols, BLOCK_V):
            offs = start + tl.arange(0, BLOCK_V)
            mask = offs < n_cols

            x = tl.load(
                logits_ptr + row * stride_logits_row + offs,
                mask=mask,
                other=0.0,
            ).to(tl.float32)

            # Recover softmax probability from saved lse (no extra memory).
            softmax_val = tl.exp(x - lse)

            # Subtract one-hot at the target column.
            is_target = (offs == label) & mask
            softmax_val = softmax_val - tl.where(is_target, 1.0, 0.0)

            dlogit = tl.where(is_valid, grad * softmax_val, 0.0)

            # Match output dtype to logits dtype; all arithmetic was in FP32.
            raw = tl.load(
                logits_ptr + row * stride_logits_row + offs,
                mask=mask,
                other=0.0,
            )
            tl.store(
                dlogits_ptr + row * stride_dlogits_row + offs,
                dlogit.to(raw.dtype),
                mask=mask,
            )


class _FusedCrossEntropyFunction(torch.autograd.Function):
    """Autograd wrapper around the Triton cross-entropy forward/backward kernels."""

    @staticmethod
    def forward(
        ctx,
        logits: torch.Tensor,
        labels: torch.Tensor,
        ignore_index: int,
    ) -> torch.Tensor:
        assert triton is not None
        n_rows, n_cols = logits.shape

        logits_c = logits.contiguous()
        labels_c = labels.contiguous().to(torch.int32)

        loss = torch.empty(n_rows, dtype=torch.float32, device=logits.device)
        lse = torch.empty(n_rows, dtype=torch.float32, device=logits.device)

        block_v = min(_BLOCK_V, triton.next_power_of_2(n_cols))
        block_v = max(block_v, 16)

        num_warps = 8 if block_v >= 512 else 4

        _ce_fwd_kernel[(n_rows,)](
            logits_c,
            labels_c,
            loss,
            lse,
            logits_c.stride(0),
            n_cols,
            ignore_index,
            BLOCK_V=block_v,
            num_warps=num_warps,
        )

        ctx.save_for_backward(logits_c, labels_c, lse)
        ctx.ignore_index = ignore_index
        ctx.block_v = block_v
        ctx.num_warps = num_warps
        return loss

    @staticmethod
    def backward(
        ctx, grad_loss: torch.Tensor
    ) -> tuple[torch.Tensor, None, None]:
        logits, labels, lse = ctx.saved_tensors
        n_rows, n_cols = logits.shape

        if not grad_loss.is_contiguous():
            grad_loss = grad_loss.contiguous()

        dlogits = torch.empty_like(logits)

        _ce_bwd_kernel[(n_rows,)](
            logits,
            labels,
            lse,
            grad_loss.to(torch.float32),
            dlogits,
            logits.stride(0),
            dlogits.stride(0),
            n_cols,
            ctx.ignore_index,
            BLOCK_V=ctx.block_v,
            num_warps=ctx.num_warps,
        )

        return dlogits, None, None


def can_use_triton_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> bool:
    """Return True when the fused Triton cross-entropy kernel can be applied."""
    if triton is None:
        return False
    if not (logits.is_cuda and labels.is_cuda):
        return False
    if logits.dim() != 2:
        return False
    if labels.dim() != 1:
        return False
    if logits.shape[0] != labels.shape[0]:
        return False
    if logits.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    return True


def triton_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute cross-entropy loss with a fused Triton kernel.

    Drop-in replacement for ``F.cross_entropy`` for the common ``[N, V]`` + ``[N]``
    case.  The kernel avoids materialising the ``[N, V]`` softmax matrix — peak extra
    memory is ``O(N)`` (the ``lse`` vector) rather than ``O(N × V)``.

    Args:
        logits:       ``[N, vocab_size]`` — raw (unnormalised) logit matrix.
        labels:       ``[N]`` — integer target token indices.
        ignore_index: Positions where ``labels == ignore_index`` contribute zero loss
                      and zero gradient.  Defaults to ``-100`` (PyTorch convention).
        reduction:    ``"mean"`` (default) or ``"sum"`` or ``"none"``.

    Returns:
        Scalar loss (``"mean"`` / ``"sum"``) or per-token loss tensor (``"none"``).

    Falls back to ``F.cross_entropy`` when Triton is unavailable or inputs are on CPU.
    """
    if not can_use_triton_cross_entropy(logits, labels):
        return F.cross_entropy(logits, labels, ignore_index=ignore_index, reduction=reduction)

    per_token = _FusedCrossEntropyFunction.apply(logits, labels, ignore_index)

    if reduction == "none":
        return per_token

    # Mean over valid (non-ignored) positions to match F.cross_entropy semantics.
    valid_mask = labels != ignore_index
    n_valid = valid_mask.sum().clamp(min=1)

    if reduction == "sum":
        return per_token.sum()
    # Default: "mean"
    return per_token.sum() / n_valid.to(per_token.dtype)
