import math

import torch


class PagedKVLayerCache:
    def __init__(
        self,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        page_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        if page_size <= 0:
            raise ValueError("page_size must be > 0")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be > 0")

        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.page_size = page_size
        self.max_pages = math.ceil(max_seq_len / page_size)
        self.length = 0

        self.key_pages = torch.empty(
            self.max_pages,
            num_kv_heads,
            page_size,
            head_dim,
            device=device,
            dtype=dtype,
        )
        self.value_pages = torch.empty_like(self.key_pages)

    def append(self, k: torch.Tensor, v: torch.Tensor) -> None:
        if k.shape != v.shape:
            raise ValueError("k and v shapes must match")
        if k.dim() != 4:
            raise ValueError("Expected k/v shape [batch, kv_heads, seq, head_dim]")
        if k.shape[0] != 1:
            raise ValueError("PagedKVLayerCache currently supports batch size 1")
        if k.shape[1] != self.num_kv_heads or k.shape[3] != self.head_dim:
            raise ValueError("k/v shapes do not match cache layout")

        seq = k.shape[2]
        if self.length + seq > self.max_seq_len:
            raise ValueError("Appending would exceed max_seq_len")

        for i in range(seq):
            t = self.length + i
            page_idx = t // self.page_size
            offset = t % self.page_size
            self.key_pages[page_idx, :, offset, :] = k[0, :, i, :]
            self.value_pages[page_idx, :, offset, :] = v[0, :, i, :]

        self.length += seq

    def get_kv(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.length == 0:
            empty = self.key_pages.new_empty((1, self.num_kv_heads, 0, self.head_dim))
            return empty, empty

        chunks_k = []
        chunks_v = []
        remaining = self.length
        for page_idx in range(self.max_pages):
            if remaining <= 0:
                break
            take = min(self.page_size, remaining)
            chunks_k.append(self.key_pages[page_idx, :, :take, :])
            chunks_v.append(self.value_pages[page_idx, :, :take, :])
            remaining -= take

        k = torch.cat(chunks_k, dim=1).unsqueeze(0)
        v = torch.cat(chunks_v, dim=1).unsqueeze(0)
        return k, v

    def clear(self) -> None:
        self.length = 0


class PagedKVCache:
    def __init__(self, layer_caches: list[PagedKVLayerCache]):
        if not layer_caches:
            raise ValueError("layer_caches must not be empty")
        self.layer_caches = layer_caches

    @classmethod
    def create(
        cls,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        page_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "PagedKVCache":
        return cls(
            [
                PagedKVLayerCache(
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    max_seq_len=max_seq_len,
                    page_size=page_size,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

    def get_layer(self, layer_idx: int) -> PagedKVLayerCache:
        return self.layer_caches[layer_idx]

    def clear(self) -> None:
        for layer in self.layer_caches:
            layer.clear()
