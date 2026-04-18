import argparse
import time

import torch

from llm_cuda.kernels.triton.attention import triton_fused_causal_attention


def torch_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    scale = 1.0 / (q.shape[-1] ** 0.5)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    seq = q.shape[-2]
    mask = torch.triu(
        torch.full((seq, seq), float("-inf"), device=q.device, dtype=q.dtype),
        diagonal=1,
    )
    probs = torch.softmax(scores + mask, dim=-1)
    return torch.matmul(probs, v)


def bench(fn, q, k, v, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        _ = fn(q, k, v)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        _ = fn(q, k, v)
    torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start) * 1000.0 / iters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Triton fused causal attention.")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"])
    return parser.parse_args()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    args = parse_args()
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    try:
        q = torch.randn(args.batch, args.heads, args.seq, args.head_dim, device="cuda", dtype=dtype)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            torch.cuda.empty_cache()
            raise RuntimeError(
                "CUDA OOM while allocating benchmark tensors. "
                "Try smaller --batch/--heads/--seq, close other GPU workloads, or use a larger GPU."
            ) from exc
        raise

    torch_ms = bench(torch_attention, q, k, v, args.warmup, args.iters)
    triton_ms = bench(triton_fused_causal_attention, q, k, v, args.warmup, args.iters)

    print(f"shape=(B={args.batch}, H={args.heads}, S={args.seq}, D={args.head_dim})")
    print(f"dtype={args.dtype}")
    print(f"torch_ms={torch_ms:.4f}")
    print(f"triton_ms={triton_ms:.4f}")
    print(f"speedup={torch_ms / triton_ms:.3f}x")


if __name__ == "__main__":
    main()
