import torch
import torch.nn as nn

from llm_cuda.kernels.cuda.rmsnorm import cuda_rms_norm_extension
from llm_cuda.kernels.triton.rmsnorm import triton_rms_norm


class RMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        use_triton: bool = True,
        use_cuda_extension: bool = True,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self.use_triton = use_triton
        self.use_cuda_extension = use_cuda_extension

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_cuda_extension and x.is_cuda:
            out = cuda_rms_norm_extension(x, self.weight, self.eps)
            if out is not None:
                return out

        if self.use_triton and x.is_cuda:
            return triton_rms_norm(x, self.weight, self.eps)

        var = x.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(var + self.eps)
        return x_norm * self.weight
