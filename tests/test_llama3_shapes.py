import torch

from llm_cuda.models.llama3.config import Llama3Config
from llm_cuda.models.llama3.kv_cache import PagedKVCache, PagedKVLayerCache
from llm_cuda.models.llama3.model import Llama3ForCausalLM


def test_llama3_forward_shapes() -> None:
    config = Llama3Config(
        vocab_size=1024,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=4,
        max_position_embeddings=256,
    )
    model = Llama3ForCausalLM(config)

    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    out = model(input_ids=input_ids)

    assert out["logits"] is not None
    assert out["logits"].shape == (2, 16, config.vocab_size)


def test_llama3_loss_path() -> None:
    config = Llama3Config(
        vocab_size=1024,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=4,
        max_position_embeddings=256,
    )
    model = Llama3ForCausalLM(config)

    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    labels = input_ids.clone()
    out = model(input_ids=input_ids, labels=labels)

    assert out["loss"] is not None
    assert out["loss"].dim() == 0


def test_llama3_kv_cache_decode_parity() -> None:
    torch.manual_seed(123)
    config = Llama3Config(
        vocab_size=1024,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=4,
        max_position_embeddings=256,
    )
    model = Llama3ForCausalLM(config)
    model.eval()

    input_ids = torch.randint(0, config.vocab_size, (1, 12))
    full = model(input_ids=input_ids)
    full_logits = full["logits"]
    assert full_logits is not None

    prefill_len = 8
    prefill = model(input_ids=input_ids[:, :prefill_len], use_cache=True)
    past = prefill["past_key_values"]
    assert past is not None

    step_logits = []
    for idx in range(prefill_len, input_ids.shape[1]):
        step = model(
            input_ids=input_ids[:, idx : idx + 1],
            past_key_values=past,
            use_cache=True,
        )
        past = step["past_key_values"]
        assert past is not None
        logits = step["logits"]
        assert logits is not None
        step_logits.append(logits)

    cached_logits = torch.cat(step_logits, dim=1)
    target_logits = full_logits[:, prefill_len:, :]

    assert cached_logits.shape == target_logits.shape
    assert torch.allclose(cached_logits, target_logits, atol=1e-5, rtol=1e-5)


def test_llama3_generate_shape() -> None:
    torch.manual_seed(124)
    config = Llama3Config(
        vocab_size=1024,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=4,
        max_position_embeddings=256,
    )
    model = Llama3ForCausalLM(config)

    prompt = torch.randint(0, config.vocab_size, (2, 6))
    generated = model.generate(prompt, max_new_tokens=4)
    assert generated.shape == (2, 10)


def test_llama3_tensor_parallel_forward_shape() -> None:
    torch.manual_seed(125)
    config = Llama3Config(
        vocab_size=1024,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=4,
        max_position_embeddings=256,
        tensor_parallel_size=2,
    )
    model = Llama3ForCausalLM(config)

    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    out = model(input_ids=input_ids)

    assert out["logits"] is not None
    assert out["logits"].shape == (2, 16, config.vocab_size)


def test_paged_kv_layer_cache_append_and_get() -> None:
    cache = PagedKVLayerCache(
        num_kv_heads=4,
        head_dim=8,
        max_seq_len=16,
        page_size=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    k = torch.randn(1, 4, 6, 8)
    v = torch.randn(1, 4, 6, 8)
    cache.append(k, v)

    k_out, v_out = cache.get_kv()
    assert cache.length == 6
    assert k_out.shape == (1, 4, 6, 8)
    assert v_out.shape == (1, 4, 6, 8)
    assert torch.allclose(k_out, k)
    assert torch.allclose(v_out, v)


def test_llama3_paged_kv_decode_parity() -> None:
    torch.manual_seed(126)
    config = Llama3Config(
        vocab_size=1024,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=4,
        max_position_embeddings=256,
    )
    model = Llama3ForCausalLM(config)
    model.eval()

    input_ids = torch.randint(0, config.vocab_size, (1, 12))
    full_logits = model(input_ids=input_ids)["logits"]
    assert full_logits is not None

    prefill_len = 8
    paged_cache = PagedKVCache.create(
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        max_seq_len=32,
        page_size=4,
        device=input_ids.device,
        dtype=model.model.embed_tokens.weight.dtype,
    )

    _ = model(input_ids=input_ids[:, :prefill_len], past_key_values=paged_cache, use_cache=True)

    step_logits = []
    for idx in range(prefill_len, input_ids.shape[1]):
        step = model(
            input_ids=input_ids[:, idx : idx + 1],
            past_key_values=paged_cache,
            use_cache=True,
        )
        logits = step["logits"]
        assert logits is not None
        step_logits.append(logits)

    cached_logits = torch.cat(step_logits, dim=1)
    target_logits = full_logits[:, prefill_len:, :]

    assert cached_logits.shape == target_logits.shape
    assert torch.allclose(cached_logits, target_logits, atol=1e-5, rtol=1e-5)
