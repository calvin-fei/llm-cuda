import argparse
import math
from dataclasses import dataclass

import torch

from llm_cuda.kernels.triton.attention import can_use_triton_fused_attention, triton_fused_causal_attention


@dataclass
class SweepResult:
    seq: int
    dtype: str
    status: str
    kernel: str
    out_max_abs: float | None
    q_grad_max_abs: float | None
    k_grad_max_abs: float | None
    v_grad_max_abs: float | None


def reference_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    seq_len = q.shape[-2]
    causal_mask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), dtype=q.dtype, device=q.device),
        diagonal=1,
    )
    probs = torch.softmax(scores + causal_mask, dim=-1)
    return torch.matmul(probs, v)


def max_abs_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    return torch.max(torch.abs(x - y)).item()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep Triton attention stability across seq lengths/dtypes")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--seq-list", type=str, default="128,256,512,1024,2048")
    parser.add_argument("--dtype-list", type=str, default="fp16,bf16")
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def run_case(batch: int, heads: int, seq: int, head_dim: int, dtype: torch.dtype, seed: int) -> SweepResult:
    dtype_name = "fp16" if dtype == torch.float16 else "bf16"
    torch.manual_seed(seed + seq)

    try:
        q_base = torch.randn(batch, heads, seq, head_dim, device="cuda", dtype=dtype)
        k_base = torch.randn_like(q_base)
        v_base = torch.randn_like(q_base)
        grad_out = torch.randn_like(q_base)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            torch.cuda.empty_cache()
            return SweepResult(seq, dtype_name, "oom", "n/a", None, None, None, None)
        raise

    kernel = "triton" if can_use_triton_fused_attention(q_base, k_base, v_base) else "fallback"

    try:
        q = q_base.clone().detach().requires_grad_(True)
        k = k_base.clone().detach().requires_grad_(True)
        v = v_base.clone().detach().requires_grad_(True)

        q_ref = q_base.clone().detach().requires_grad_(True)
        k_ref = k_base.clone().detach().requires_grad_(True)
        v_ref = v_base.clone().detach().requires_grad_(True)

        out = triton_fused_causal_attention(q, k, v)
        out_ref = reference_attention(q_ref, k_ref, v_ref)

        (out * grad_out).sum().backward()
        (out_ref * grad_out).sum().backward()
    except Exception as exc:  # pragma: no cover - depends on runtime GPU pressure
        if "out of memory" in str(exc).lower():
            torch.cuda.empty_cache()
            return SweepResult(seq, dtype_name, "oom", kernel, None, None, None, None)
        raise

    return SweepResult(
        seq=seq,
        dtype=dtype_name,
        status="ok",
        kernel=kernel,
        out_max_abs=max_abs_diff(out, out_ref),
        q_grad_max_abs=max_abs_diff(q.grad, q_ref.grad),
        k_grad_max_abs=max_abs_diff(k.grad, k_ref.grad),
        v_grad_max_abs=max_abs_diff(v.grad, v_ref.grad),
    )


def print_results(results: list[SweepResult]) -> None:
    print("seq,dtype,status,kernel,out_max_abs,q_grad_max_abs,k_grad_max_abs,v_grad_max_abs")
    for r in results:
        print(
            f"{r.seq},{r.dtype},{r.status},{r.kernel},"
            f"{'' if r.out_max_abs is None else f'{r.out_max_abs:.6e}'},"
            f"{'' if r.q_grad_max_abs is None else f'{r.q_grad_max_abs:.6e}'},"
            f"{'' if r.k_grad_max_abs is None else f'{r.k_grad_max_abs:.6e}'},"
            f"{'' if r.v_grad_max_abs is None else f'{r.v_grad_max_abs:.6e}'}"
        )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for stability sweep")

    args = parse_args()
    seq_list = [int(x.strip()) for x in args.seq_list.split(",") if x.strip()]
    dtype_list_raw = [x.strip().lower() for x in args.dtype_list.split(",") if x.strip()]

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16}
    dtype_list = [dtype_map[x] for x in dtype_list_raw]

    results: list[SweepResult] = []
    for seq in seq_list:
        for dtype in dtype_list:
            results.append(
                run_case(
                    batch=args.batch,
                    heads=args.heads,
                    seq=seq,
                    head_dim=args.head_dim,
                    dtype=dtype,
                    seed=args.seed,
                )
            )

    print_results(results)


if __name__ == "__main__":
    main()
