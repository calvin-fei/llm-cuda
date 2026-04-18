import argparse
import time

import torch

from llm_cuda.kernels.triton.swiglu import triton_swiglu


def torch_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.silu(gate) * up


def bench(fn, gate, up, warmup: int, iters: int) -> float:
    try:
        for _ in range(warmup):
            _ = fn(gate, up)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(iters):
            _ = fn(gate, up)
        torch.cuda.synchronize()
        end = time.perf_counter()
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            torch.cuda.empty_cache()
            raise RuntimeError(
                "CUDA OOM during benchmark execution. "
                "Try smaller --batch/--seq/--hidden, close other GPU workloads, or use a larger GPU."
            ) from exc
        raise

    return (end - start) * 1000.0 / iters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Triton fused SwiGLU")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=4096)
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
        gate = torch.randn(args.batch, args.seq, args.hidden, device="cuda", dtype=dtype)
        up = torch.randn_like(gate)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            torch.cuda.empty_cache()
            raise RuntimeError(
                "CUDA OOM while allocating benchmark tensors. "
                "Try smaller --batch/--seq/--hidden, close other GPU workloads, or use a larger GPU."
            ) from exc
        raise

    torch_ms = bench(torch_swiglu, gate, up, args.warmup, args.iters)
    triton_ms = bench(triton_swiglu, gate, up, args.warmup, args.iters)

    print(f"shape=(B={args.batch}, S={args.seq}, H={args.hidden})")
    print(f"dtype={args.dtype}")
    print(f"torch_ms={torch_ms:.4f}")
    print(f"triton_ms={triton_ms:.4f}")
    print(f"speedup={torch_ms / triton_ms:.3f}x")


if __name__ == "__main__":
    main()
