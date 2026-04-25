from dataclasses import dataclass


@dataclass
class Llama3Config:
    vocab_size: int = 128256
    hidden_size: int = 1024
    intermediate_size: int = 2816
    num_hidden_layers: int = 12
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    max_position_embeddings: int = 8192
    rope_theta: float = 500000.0
    rms_norm_eps: float = 1e-5
    initializer_range: float = 0.02
    tensor_parallel_size: int = 1
    tie_word_embeddings: bool = False
    pad_token_id: int = 0
    bos_token_id: int = 128000
    eos_token_id: int = 128001
    gradient_checkpointing: bool = False

    def __post_init__(self) -> None:
        if self.tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be >= 1")

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")

        if self.hidden_size % self.tensor_parallel_size != 0:
            raise ValueError("hidden_size must be divisible by tensor_parallel_size")

        if self.intermediate_size % self.tensor_parallel_size != 0:
            raise ValueError("intermediate_size must be divisible by tensor_parallel_size")

        kv_size = self.num_key_value_heads * self.head_dim
        if kv_size % self.tensor_parallel_size != 0:
            raise ValueError("num_key_value_heads * head_dim must be divisible by tensor_parallel_size")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads
