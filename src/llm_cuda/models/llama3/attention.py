import torch
import torch.nn as nn

from llm_cuda.kernels.triton.attention import triton_fused_causal_attention
from llm_cuda.kernels.triton.decode_attention import (
    can_use_triton_decode_attention,
    triton_decode_attention,
)
from llm_cuda.parallel.tensor_parallel import ColumnParallelLinear, RowParallelLinear

from .config import Llama3Config
from .kv_cache import PagedKVLayerCache
from .rotary import apply_rotary, build_inv_freq, build_rope_cache


class Llama3Attention(nn.Module):
    def __init__(self, config: Llama3Config):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        tp_size = config.tensor_parallel_size

        if tp_size > 1:
            self.q_proj = ColumnParallelLinear(config.hidden_size, q_size, tp_size=tp_size, bias=False)
            self.k_proj = ColumnParallelLinear(config.hidden_size, kv_size, tp_size=tp_size, bias=False)
            self.v_proj = ColumnParallelLinear(config.hidden_size, kv_size, tp_size=tp_size, bias=False)
            self.o_proj = RowParallelLinear(q_size, config.hidden_size, tp_size=tp_size, bias=False)
        else:
            self.q_proj = nn.Linear(config.hidden_size, q_size, bias=False)
            self.k_proj = nn.Linear(config.hidden_size, kv_size, bias=False)
            self.v_proj = nn.Linear(config.hidden_size, kv_size, bias=False)
            self.o_proj = nn.Linear(q_size, config.hidden_size, bias=False)

        # Pre-compute inv_freq once and register as a non-persistent buffer so
        # it moves with the module to the correct device but is not saved in
        # state_dict checkpoints.  This avoids recomputing arange + pow on
        # every forward step (critical during decode where one token is
        # processed per step).
        inv_freq = build_inv_freq(
            head_dim=config.head_dim,
            base=config.rope_theta,
            device=torch.device("cpu"),
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _shape(self, x: torch.Tensor, heads: int) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, heads, self.head_dim).transpose(1, 2)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_kv_groups == 1:
            return x
        bsz, n_kv, seq_len, head_dim = x.shape
        x = x[:, :, None, :, :].expand(bsz, n_kv, self.num_kv_groups, seq_len, head_dim)
        return x.reshape(bsz, n_kv * self.num_kv_groups, seq_len, head_dim)

    def _torch_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_len: int,
    ) -> torch.Tensor:
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)

        q_len = q.shape[-2]
        kv_len = k.shape[-2]
        q_pos = torch.arange(past_len, past_len + q_len, device=q.device).unsqueeze(1)
        k_pos = torch.arange(kv_len, device=q.device).unsqueeze(0)
        causal = k_pos <= q_pos
        causal_mask = torch.where(
            causal,
            torch.zeros((q_len, kv_len), dtype=q.dtype, device=q.device),
            torch.full((q_len, kv_len), float("-inf"), dtype=q.dtype, device=q.device),
        )
        attn_scores = attn_scores + causal_mask.unsqueeze(0).unsqueeze(0)

        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_probs = torch.softmax(attn_scores, dim=-1)
        return torch.matmul(attn_probs, v)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | PagedKVLayerCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | PagedKVLayerCache | None]:
        bsz, seq_len, _ = x.shape

        q = self._shape(self.q_proj(x), self.num_heads)
        k = self._shape(self.k_proj(x), self.num_kv_heads)
        v = self._shape(self.v_proj(x), self.num_kv_heads)

        past_len = 0
        if isinstance(past_key_value, PagedKVLayerCache):
            past_len = past_key_value.length
        elif past_key_value is not None:
            past_len = past_key_value[0].shape[-2]

        cos, sin = build_rope_cache(
            seq_len=seq_len,
            head_dim=self.head_dim,
            base=self.config.rope_theta,
            device=x.device,
            dtype=x.dtype,
            position_offset=past_len,
            inv_freq=self.inv_freq,
        )
        q, k = apply_rotary(q, k, cos, sin)

        if isinstance(past_key_value, PagedKVLayerCache):
            if use_cache:
                past_key_value.append(k, v)
            k, v = past_key_value.get_kv()
        elif past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)

        if isinstance(past_key_value, PagedKVLayerCache):
            present = past_key_value if use_cache else None
        else:
            present = (k, v) if use_cache else None

        # Decode fast-path: single query token + Triton flash-decode kernel.
        # Handled before _repeat_kv so K/V are accessed at their native
        # (unexpanded) n_kv_heads count, saving 2× memory bandwidth for GQA.
        if q.shape[-2] == 1 and can_use_triton_decode_attention(q, k, v, attention_mask):
            out = triton_decode_attention(q, k, v)
        else:
            k = self._repeat_kv(k)
            v = self._repeat_kv(v)

            if past_len == 0 and q.shape[-2] == k.shape[-2] and attention_mask is None:
                out = triton_fused_causal_attention(q, k, v, attention_mask=attention_mask)
            else:
                out = self._torch_attention(q, k, v, attention_mask=attention_mask, past_len=past_len)

        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.num_heads * self.head_dim)
        return self.o_proj(out), present
