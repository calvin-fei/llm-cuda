import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ColumnParallelLinear(nn.Module):
    """
    Splits out_features across tensor-parallel shards and concatenates outputs.
    """

    def __init__(self, in_features: int, out_features: int, tp_size: int, bias: bool = False):
        super().__init__()
        if tp_size < 1:
            raise ValueError("tp_size must be >= 1")
        if out_features % tp_size != 0:
            raise ValueError("out_features must be divisible by tp_size")

        self.in_features = in_features
        self.out_features = out_features
        self.tp_size = tp_size
        self.out_per_shard = out_features // tp_size

        self.weight_shards = nn.ParameterList(
            [nn.Parameter(torch.empty(self.out_per_shard, in_features)) for _ in range(tp_size)]
        )
        self.bias_shards = None
        if bias:
            self.bias_shards = nn.ParameterList(
                [nn.Parameter(torch.empty(self.out_per_shard)) for _ in range(tp_size)]
            )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.in_features)
        for w in self.weight_shards:
            nn.init.uniform_(w, -bound, bound)
        if self.bias_shards is not None:
            for b in self.bias_shards:
                nn.init.uniform_(b, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for idx, w in enumerate(self.weight_shards):
            b = None if self.bias_shards is None else self.bias_shards[idx]
            outputs.append(F.linear(x, w, b))
        return torch.cat(outputs, dim=-1)


class RowParallelLinear(nn.Module):
    """
    Splits in_features across tensor-parallel shards and sums partial outputs.
    """

    def __init__(self, in_features: int, out_features: int, tp_size: int, bias: bool = False):
        super().__init__()
        if tp_size < 1:
            raise ValueError("tp_size must be >= 1")
        if in_features % tp_size != 0:
            raise ValueError("in_features must be divisible by tp_size")

        self.in_features = in_features
        self.out_features = out_features
        self.tp_size = tp_size
        self.in_per_shard = in_features // tp_size

        self.weight_shards = nn.ParameterList(
            [nn.Parameter(torch.empty(out_features, self.in_per_shard)) for _ in range(tp_size)]
        )
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.in_per_shard)
        for w in self.weight_shards:
            nn.init.uniform_(w, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunks = torch.chunk(x, self.tp_size, dim=-1)
        out = F.linear(chunks[0], self.weight_shards[0], None)
        for idx in range(1, self.tp_size):
            out = out + F.linear(chunks[idx], self.weight_shards[idx], None)
        if self.bias is not None:
            out = out + self.bias
        return out
