"""Tensor-parallel building blocks."""

from .tensor_parallel import ColumnParallelLinear, RowParallelLinear

__all__ = ["ColumnParallelLinear", "RowParallelLinear"]
