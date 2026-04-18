import torch
import torch.nn as nn

from llm_cuda.parallel.tensor_parallel import ColumnParallelLinear, RowParallelLinear

from .attention import Llama3Attention
from .config import Llama3Config
from .kv_cache import PagedKVCache, PagedKVLayerCache
from .mlp import Llama3MLP
from .norm import RMSNorm


class Llama3DecoderLayer(nn.Module):
    def __init__(self, config: Llama3Config):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Llama3Attention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = Llama3MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | PagedKVLayerCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | PagedKVLayerCache | None]:
        residual = x
        x = self.input_layernorm(x)
        x, present = self.self_attn(
            x,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        x = x + residual

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = x + residual
        return x, present


class Llama3Model(nn.Module):
    def __init__(self, config: Llama3Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Llama3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | PagedKVCache | None = None,
        use_cache: bool = False,
    ) -> dict[str, torch.Tensor | list[tuple[torch.Tensor, torch.Tensor]] | PagedKVCache | None]:
        x = self.embed_tokens(input_ids)
        next_past_key_values: list[tuple[torch.Tensor, torch.Tensor]] = []

        for idx, layer in enumerate(self.layers):
            if isinstance(past_key_values, PagedKVCache):
                layer_past = past_key_values.get_layer(idx)
            else:
                layer_past = None if past_key_values is None else past_key_values[idx]
            x, present = layer(
                x,
                attention_mask=attention_mask,
                past_key_value=layer_past,
                use_cache=use_cache,
            )
            if use_cache and present is not None and not isinstance(past_key_values, PagedKVCache):
                next_past_key_values.append(present)
        x = self.norm(x)

        if isinstance(past_key_values, PagedKVCache):
            next_cache: list[tuple[torch.Tensor, torch.Tensor]] | PagedKVCache | None = (
                past_key_values if use_cache else None
            )
        else:
            next_cache = next_past_key_values if use_cache else None

        return {
            "hidden_states": x,
            "past_key_values": next_cache,
        }


class Llama3ForCausalLM(nn.Module):
    def __init__(self, config: Llama3Config):
        super().__init__()
        self.config = config
        self.model = Llama3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, ColumnParallelLinear):
            for w in module.weight_shards:
                nn.init.normal_(w, mean=0.0, std=self.config.initializer_range)
            if module.bias_shards is not None:
                for b in module.bias_shards:
                    nn.init.zeros_(b)
        elif isinstance(module, RowParallelLinear):
            for w in module.weight_shards:
                nn.init.normal_(w, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | PagedKVCache | None = None,
        use_cache: bool = False,
    ) -> dict[str, torch.Tensor | list[tuple[torch.Tensor, torch.Tensor]] | PagedKVCache | None]:
        model_out = self.model(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        hidden_states = model_out["hidden_states"]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {
            "logits": logits,
            "loss": loss,
            "past_key_values": model_out["past_key_values"],
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 16,
        use_paged_kv: bool = False,
        page_size: int = 64,
    ) -> torch.Tensor:
        self.eval()

        generated = input_ids
        if use_paged_kv:
            past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | PagedKVCache | None = PagedKVCache.create(
                num_layers=self.config.num_hidden_layers,
                num_kv_heads=self.config.num_key_value_heads,
                head_dim=self.config.head_dim,
                max_seq_len=input_ids.shape[1] + max_new_tokens,
                page_size=page_size,
                device=input_ids.device,
                dtype=self.model.embed_tokens.weight.dtype,
            )
        else:
            past_key_values = None
        cur_input = input_ids

        for _ in range(max_new_tokens):
            out = self.forward(
                input_ids=cur_input,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = out["logits"]
            past_key_values = out["past_key_values"]
            if logits is None:
                raise RuntimeError("Expected logits during generation")

            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            cur_input = next_token

        return generated
