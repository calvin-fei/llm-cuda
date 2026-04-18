import argparse

import torch

from llm_cuda.models.llama3.config import Llama3Config
from llm_cuda.models.llama3.model import Llama3ForCausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal Llama3 forward/backward pass.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--decode-steps", type=int, default=0)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--paged-kv", action="store_true")
    parser.add_argument("--page-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = Llama3Config(
        vocab_size=args.vocab_size,
        hidden_size=1024,
        intermediate_size=2816,
        num_hidden_layers=8,
        num_attention_heads=16,
        num_key_value_heads=8,
        tensor_parallel_size=args.tp_size,
    )
    model = Llama3ForCausalLM(config).to(args.device)
    model.train()

    input_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(args.batch_size, args.seq_len),
        device=args.device,
    )
    labels = input_ids.clone()

    output = model(input_ids=input_ids, labels=labels)
    loss = output["loss"]
    if loss is None:
        raise RuntimeError("Expected a loss when labels are provided")

    loss.backward()

    print(f"device={args.device}")
    print(f"logits_shape={tuple(output['logits'].shape)}")
    print(f"loss={loss.item():.6f}")

    if args.decode_steps > 0:
        model.eval()
        prompt = torch.randint(
            low=0,
            high=config.vocab_size,
            size=(args.batch_size, 8),
            device=args.device,
        )
        if args.paged_kv:
            generated = model.generate(
                prompt,
                max_new_tokens=args.decode_steps,
                use_paged_kv=True,
                page_size=args.page_size,
            )
        else:
            generated = model.generate(prompt, max_new_tokens=args.decode_steps)
        print(f"generated_shape={tuple(generated.shape)}")


if __name__ == "__main__":
    main()
