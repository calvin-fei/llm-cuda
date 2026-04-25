import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _rmsnorm_kernel(
        x_ptr,
        w_ptr,
        y_ptr,
        stride_xm,
        stride_ym,
        hidden_size,
        eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < hidden_size

        x_row_ptr = x_ptr + row * stride_xm + cols
        w_ptrs = w_ptr + cols
        y_row_ptr = y_ptr + row * stride_ym + cols

        x = tl.load(x_row_ptr, mask=mask, other=0.0)
        w = tl.load(w_ptrs, mask=mask, other=1.0)

        # Upcast to float32 for the variance accumulation and scale application so that
        # FP16 inputs don't overflow during x * x or lose precision in the reduction.
        x_f32 = x.to(tl.float32)
        w_f32 = w.to(tl.float32)
        var = tl.sum(x_f32 * x_f32, axis=0) / hidden_size
        inv_rms = tl.rsqrt(var + eps)
        y = x_f32 * inv_rms * w_f32

        tl.store(y_row_ptr, y.to(x.dtype), mask=mask)


def triton_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    if triton is None or not x.is_cuda:
        var = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(var + eps) * weight

    if x.dim() != 3:
        raise ValueError("Expected x shape [batch, seq, hidden]")

    bsz, seq_len, hidden = x.shape
    x_2d = x.reshape(bsz * seq_len, hidden)
    y_2d = torch.empty_like(x_2d)

    block_size = triton.next_power_of_2(hidden)
    block_size = min(max(block_size, 128), 4096)
    num_warps = 8 if block_size >= 2048 else 4

    grid = (x_2d.shape[0],)
    _rmsnorm_kernel[grid](
        x_2d,
        weight,
        y_2d,
        x_2d.stride(0),
        y_2d.stride(0),
        hidden,
        eps,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )

    return y_2d.view(bsz, seq_len, hidden)
