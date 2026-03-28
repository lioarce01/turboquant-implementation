"""
KV Cache Hook for HuggingFace Transformers (standard attention models).
Subclasses DynamicCache to compress KV pairs with TurboQuant.

Tested with Qwen2.5-1.5B-Instruct (pure transformer, all layers use KV cache).

Usage:
    cache = TurboQuantCache(bits=4)
    output = model.generate(**inputs, past_key_values=cache, use_cache=True)
    print(cache.stats.report())
"""

import torch
import math
import time
from typing import Optional, Tuple, Dict
from dataclasses import dataclass

from transformers import PreTrainedModel
from transformers.cache_utils import DynamicCache

from .turboquant import TurboQuantMSE


# ---------------------------------------------------------------------------
# Stats tracking
# ---------------------------------------------------------------------------

@dataclass
class QuantizationStats:
    total_tokens_compressed: int = 0
    total_compress_time_ms: float = 0.0
    total_decompress_time_ms: float = 0.0
    original_bytes: int = 0
    compressed_bytes: int = 0

    @property
    def compress_ratio(self) -> float:
        if self.compressed_bytes == 0:
            return 0.0
        return self.original_bytes / self.compressed_bytes

    @property
    def avg_compress_ms_per_token(self) -> float:
        if self.total_tokens_compressed == 0:
            return 0.0
        return self.total_compress_time_ms / self.total_tokens_compressed

    def report(self) -> dict:
        return {
            "total_tokens_compressed": self.total_tokens_compressed,
            "compress_ratio": round(self.compress_ratio, 2),
            "original_mb": round(self.original_bytes / 1e6, 2),
            "compressed_mb": round(self.compressed_bytes / 1e6, 2),
            "avg_compress_ms_per_token": round(self.avg_compress_ms_per_token, 4),
            "total_compress_time_ms": round(self.total_compress_time_ms, 2),
            "total_decompress_time_ms": round(self.total_decompress_time_ms, 2),
        }


# ---------------------------------------------------------------------------
# TurboQuantCache
# ---------------------------------------------------------------------------

class TurboQuantCache(DynamicCache):
    """
    Drop-in replacement for DynamicCache that compresses KV tensors on write
    and decompresses on read using TurboQuant.

    Instead of storing fp16 (B, H, L, D) tensors, stores:
      - int16 centroid indices  (B, H, L, padded_D)
      - fp16 norms              (B, H, L, 1)

    The full-precision KV tensor is never held in GPU memory between steps.
    """

    def __init__(self, bits: int = 4, dtype: torch.dtype = torch.float16):
        super().__init__()
        self.bits = bits
        self.dtype = dtype

        # Lazy-initialized quantizers keyed by head_dim
        self._quantizers: Dict[int, TurboQuantMSE] = {}

        # Compressed storage: layer_idx -> (indices int16, norms fp16)
        self._key_compressed: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._val_compressed: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

        self.stats = QuantizationStats()

    def _get_quantizer(self, head_dim: int, device: torch.device) -> TurboQuantMSE:
        if head_dim not in self._quantizers:
            self._quantizers[head_dim] = TurboQuantMSE(
                dim=head_dim, bits=self.bits, device=device, dtype=self.dtype
            )
        return self._quantizers[head_dim]

    def _compress(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, H, L, D = tensor.shape
        quantizer = self._get_quantizer(D, tensor.device)

        t0 = time.perf_counter()
        indices, norms = quantizer.quantize(tensor)
        self.stats.total_compress_time_ms += (time.perf_counter() - t0) * 1000

        self.stats.total_tokens_compressed += B * H * L
        self.stats.original_bytes += B * H * L * D * 2                  # fp16 = 2 bytes
        self.stats.compressed_bytes += (                                  # actual tensor sizes
            indices.element_size() * indices.numel()
            + norms.element_size() * norms.numel()
        )
        return indices, norms

    def _decompress(self, indices: torch.Tensor, norms: torch.Tensor, head_dim: int) -> torch.Tensor:
        quantizer = self._get_quantizer(head_dim, indices.device)
        t0 = time.perf_counter()
        result = quantizer.dequantize(indices, norms)
        self.stats.total_decompress_time_ms += (time.perf_counter() - t0) * 1000
        return result

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        head_dim = key_states.shape[-1]

        if layer_idx not in self._key_compressed:
            k_idx, k_norms = self._compress(key_states)
            v_idx, v_norms = self._compress(value_states)
        else:
            old_k_idx, old_k_norms = self._key_compressed[layer_idx]
            old_v_idx, old_v_norms = self._val_compressed[layer_idx]
            new_k_idx, new_k_norms = self._compress(key_states)
            new_v_idx, new_v_norms = self._compress(value_states)
            k_idx = torch.cat([old_k_idx, new_k_idx], dim=2)
            k_norms = torch.cat([old_k_norms, new_k_norms], dim=2)
            v_idx = torch.cat([old_v_idx, new_v_idx], dim=2)
            v_norms = torch.cat([old_v_norms, new_v_norms], dim=2)

        self._key_compressed[layer_idx] = (k_idx, k_norms)
        self._val_compressed[layer_idx] = (v_idx, v_norms)

        keys = self._decompress(k_idx, k_norms, head_dim)
        values = self._decompress(v_idx, v_norms, head_dim)
        return keys, values

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx in self._key_compressed:
            return self._key_compressed[layer_idx][0].shape[2]
        return 0

    def vram_usage_bytes(self) -> int:
        total = 0
        for store in (self._key_compressed, self._val_compressed):
            for idx, norms in store.values():
                total += idx.element_size() * idx.numel()
                total += norms.element_size() * norms.numel()
        return total


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def apply_turboquant_to_model(
    model: PreTrainedModel,
    bits: int = 4,
    verbose: bool = True,
) -> TurboQuantCache:
    dtype = next(model.parameters()).dtype
    cache = TurboQuantCache(bits=bits, dtype=dtype)
    if verbose:
        n_layers = model.config.num_hidden_layers
        print(f"[TurboQuant] {bits}-bit compression on all {n_layers} attention layers")
    return cache


def run_with_cache(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    max_new_tokens: int = 64,
    use_turboquant: bool = True,
    bits: int = 4,
    **generate_kwargs,
) -> Tuple[torch.Tensor, Optional[TurboQuantCache], dict]:
    generate_args = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        **generate_kwargs,
    )

    cache = None
    if use_turboquant:
        cache = apply_turboquant_to_model(model, bits=bits, verbose=False)
        generate_args["past_key_values"] = cache

    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(**generate_args)
    total_ms = (time.perf_counter() - t0) * 1000

    n_new = output_ids.shape[1] - input_ids.shape[1]
    timing = {
        "total_ms": round(total_ms, 1),
        "tokens_per_sec": round(n_new / (total_ms / 1000), 1),
        "n_new_tokens": n_new,
    }
    return output_ids, cache, timing
