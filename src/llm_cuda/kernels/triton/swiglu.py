import torch

from llm_cuda.kernels.cuda.swiglu import cuda_swiglu_extension

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _swiglu_fwd_kernel(
        gate_ptr,
        up_ptr,
        out_ptr,
        stride_gm,
        stride_um,
        stride_om,
        hidden_size,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < hidden_size

        gate_row = gate_ptr + row * stride_gm + cols
        up_row = up_ptr + row * stride_um + cols
        out_row = out_ptr + row * stride_om + cols

        g = tl.load(gate_row, mask=mask, other=0.0)
        u = tl.load(up_row, mask=mask, other=0.0)

        g_fp32 = g.to(tl.float32)
        u_fp32 = u.to(tl.float32)
        sig = 1.0 / (1.0 + tl.exp(-g_fp32))
        y = (g_fp32 * sig) * u_fp32

        tl.store(out_row, y.to(g.dtype), mask=mask)


class _SwiGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        if gate.shape != up.shape:
            raise ValueError("gate and up must have matching shapes")

        if gate.is_cuda and up.is_cuda:
            out_cuda = cuda_swiglu_extension(gate, up)
            if out_cuda is not None:
                ctx.save_for_backward(gate, up)
                return out_cuda

        if triton is None or not gate.is_cuda or not up.is_cuda:
            sig = torch.sigmoid(gate)
            out = (gate * sig) * up
            ctx.save_for_backward(gate, up)
            return out

        if gate.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            sig = torch.sigmoid(gate)
            out = (gate * sig) * up
            ctx.save_for_backward(gate, up)
            return out

        if not gate.is_contiguous():
            gate = gate.contiguous()
        if not up.is_contiguous():
            up = up.contiguous()

        orig_shape = gate.shape
        hidden_size = orig_shape[-1]
        gate_2d = gate.view(-1, hidden_size)
        up_2d = up.view(-1, hidden_size)
        out_2d = torch.empty_like(gate_2d)

        block_size = triton.next_power_of_2(hidden_size)
        block_size = min(max(block_size, 128), 4096)
        num_warps = 8 if block_size >= 2048 else 4

        grid = (gate_2d.shape[0],)
        _swiglu_fwd_kernel[grid](
            gate_2d,
            up_2d,
            out_2d,
            gate_2d.stride(0),
            up_2d.stride(0),
            out_2d.stride(0),
            hidden_size,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )

        out = out_2d.view(orig_shape)
        ctx.save_for_backward(gate, up)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate, up = ctx.saved_tensors
        sig = torch.sigmoid(gate)
        dsilu = sig * (1.0 + gate * (1.0 - sig))

        dgate = grad_out * up * dsilu
        dup = grad_out * (gate * sig)
        return dgate, dup


def triton_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return _SwiGLUFunction.apply(gate, up)
