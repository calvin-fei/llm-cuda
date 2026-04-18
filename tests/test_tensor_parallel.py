import torch
import torch.nn.functional as F

from llm_cuda.parallel.tensor_parallel import ColumnParallelLinear, RowParallelLinear


def test_column_parallel_linear_matches_dense() -> None:
    torch.manual_seed(30)
    x = torch.randn(2, 3, 8)
    full_w = torch.randn(6, 8)

    tp = ColumnParallelLinear(in_features=8, out_features=6, tp_size=2, bias=False)
    with torch.no_grad():
        tp.weight_shards[0].copy_(full_w[:3, :])
        tp.weight_shards[1].copy_(full_w[3:, :])

    out_tp = tp(x)
    out_ref = F.linear(x, full_w)

    assert out_tp.shape == out_ref.shape
    assert torch.allclose(out_tp, out_ref, atol=1e-6, rtol=1e-6)


def test_row_parallel_linear_matches_dense() -> None:
    torch.manual_seed(31)
    x = torch.randn(2, 3, 8)
    full_w = torch.randn(5, 8)

    tp = RowParallelLinear(in_features=8, out_features=5, tp_size=2, bias=False)
    with torch.no_grad():
        tp.weight_shards[0].copy_(full_w[:, :4])
        tp.weight_shards[1].copy_(full_w[:, 4:])

    out_tp = tp(x)
    out_ref = F.linear(x, full_w)

    assert out_tp.shape == out_ref.shape
    assert torch.allclose(out_tp, out_ref, atol=1e-6, rtol=1e-6)
