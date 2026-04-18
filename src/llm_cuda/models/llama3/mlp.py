import torch
import torch.nn as nn

from llm_cuda.kernels.triton.swiglu import triton_swiglu
from llm_cuda.parallel.tensor_parallel import ColumnParallelLinear, RowParallelLinear

from .config import Llama3Config


class Llama3MLP(nn.Module):
    def __init__(self, config: Llama3Config):
        super().__init__()
        tp_size = config.tensor_parallel_size

        if tp_size > 1:
            self.gate_proj = ColumnParallelLinear(
                config.hidden_size,
                config.intermediate_size,
                tp_size=tp_size,
                bias=False,
            )
            self.up_proj = ColumnParallelLinear(
                config.hidden_size,
                config.intermediate_size,
                tp_size=tp_size,
                bias=False,
            )
            self.down_proj = RowParallelLinear(
                config.intermediate_size,
                config.hidden_size,
                tp_size=tp_size,
                bias=False,
            )
        else:
            self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
            self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
            self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        hidden = triton_swiglu(gate, up)
        return self.down_proj(hidden)
