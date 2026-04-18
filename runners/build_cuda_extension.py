import argparse

from llm_cuda.kernels.cuda.rmsnorm import cuda_extension_status
from llm_cuda.kernels.cuda.extension import load_cuda_extension


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build/load CUDA C++ extension kernels")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose build logs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ext = load_cuda_extension(verbose=args.verbose)

    if ext is None:
        print("cuda_extension=unavailable")
        print(f"status={cuda_extension_status()}")
        return

    print("cuda_extension=loaded")
    symbols = []
    if hasattr(ext, "rms_norm_forward"):
        symbols.append("rms_norm_forward")
    if hasattr(ext, "swiglu_forward"):
        symbols.append("swiglu_forward")
    if hasattr(ext, "attention_forward"):
        symbols.append("attention_forward")
    print(f"available_symbols={','.join(symbols)}")


if __name__ == "__main__":
    main()
