# llm-cuda

CUDA and Triton reimplementation project for modern LLMs and LMMs.

Initial target implemented: Llama 3 (training/inference scaffold + kernel hooks).

## Current status: Llama 3 starter

This repo now includes:

- Llama 3 style decoder-only transformer in PyTorch
- Grouped-query attention (GQA)
- RoPE (rotary embeddings)
- RMSNorm with Triton kernel entry point + PyTorch fallback
- Initial fused Triton causal-attention forward and backward kernels (with safe fallback)
- Fused Triton SwiGLU kernel with autograd-safe backward
- KV-cache support for autoregressive decoding
- Paged KV-cache option for long-running decode workloads
- Tensor-parallel sharding for QKV and MLP projections
- Optional CUDA C++ extension path for RMSNorm, SwiGLU forward, and causal attention forward (with Triton/PyTorch fallback)
- Long-context/high-concurrency optimization: automatic CUDA SDPA backend selection when beneficial
- Causal LM head + loss path
- A minimal runner for forward/backward on CPU or GPU
- Basic shape/loss tests

## Project layout

```text
src/llm_cuda/
	models/llama3/
		config.py
		attention.py
		rotary.py
		norm.py
		mlp.py
		model.py
	kernels/
		triton/attention.py
		triton/rmsnorm.py
		triton/swiglu.py
		cuda/rmsnorm.py
runners/run_llama3.py
runners/bench_triton_attention.py
runners/bench_triton_swiglu.py
runners/sweep_triton_attention_stability.py
tests/test_llama3_shapes.py
```

## Quick start

Create environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Run a minimal Llama 3 forward/backward pass:

```bash
python runners/run_llama3.py --device cuda --batch-size 2 --seq-len 128 --vocab-size 32000
```

Run decode with KV-cache:

```bash
python runners/run_llama3.py --device cuda --batch-size 1 --seq-len 64 --vocab-size 32000 --decode-steps 16
```

Run decode with paged KV-cache:

```bash
python runners/run_llama3.py --device cuda --batch-size 1 --seq-len 64 --vocab-size 32000 --decode-steps 16 --paged-kv --page-size 64
```

Run with tensor parallel sharding (single-process simulation):

```bash
python runners/run_llama3.py --device cuda --batch-size 1 --seq-len 64 --vocab-size 32000 --tp-size 2
```

If CUDA is unavailable:

```bash
python runners/run_llama3.py --device cpu
```

Run tests:

```bash
pytest -q
```

Benchmark fused Triton attention:

```bash
python runners/bench_triton_attention.py --batch 4 --heads 16 --seq 512 --head-dim 64 --dtype fp16
```

Benchmark fused Triton SwiGLU:

```bash
python runners/bench_triton_swiglu.py --batch 8 --seq 512 --hidden 4096 --dtype fp16
```

Build/load CUDA extension kernels:

```bash
python runners/build_cuda_extension.py --verbose
```

Run mixed precision stability sweep (forward + backward parity):

```bash
python runners/sweep_triton_attention_stability.py --batch 1 --heads 4 --head-dim 64 --seq-list 128,256,512,1024,2048 --dtype-list fp16,bf16
```

## Next milestones for Llama 3

1. Add optimized CUDA extension implementations and benchmarking for attention/SwiGLU backward.
2. Checkpoint conversion from Hugging Face Llama 3 weights.

## Notes

- The current implementation is intentionally compact to enable iterative kernel replacement.
- Use this as the base to progressively swap PyTorch ops with custom Triton/CUDA kernels while preserving test coverage.
