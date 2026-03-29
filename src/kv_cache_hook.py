"""
KV Cache Hook for HuggingFace Transformers (standard attention models).
Subclasses DynamicCache to compress KV pairs with TurboQuant.

Performance design
------------------
Naive implementation compresses on write and decompresses the FULL cache on
every decode step — O(N) decompression per step, O(N²) total.

This implementation uses incremental decompression:
  - On each update() call, compress the NEW tokens (O(L) per step, L=1 for decode).
  - Decompress only the NEW tokens and write to a pre-allocated fp16 working
    buffer at the next position — O(1) per decode step.
  - Return a view of the fp16 buffer — no copy.

This reduces decompression cost from O(N²) to O(N) total across all steps.
The fp16 working buffer uses the same memory as a standard DynamicCache during
generation; after generation it can be freed (call free_working_memory()) so
only the compressed representation remains.

Memory during generation:
  fp16 working buffer (for attention) + compressed storage
  = 1x fp16 + (bits/16)x fp16 ≈ 1.25x for 4-bit

Memory after free_working_memory():
  compressed storage only = (bits/16)x fp16 ≈ 0.25x for 4-bit

Usage:
    cache = TurboQuantCache(bits=4)
    output = model.generate(**inputs, past_key_values=cache, use_cache=True)
    cache.free_working_memory()   # reclaim fp16 working buffer
    print(cache.stats.report())
"""

import torch
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
    Drop-in replacement for DynamicCache that compresses KV tensors with
    TurboQuant using incremental decompression for O(1) per decode step.

    Storage layout per layer:
      _key_packed   (B, H, capacity, packed_D)  uint8  — compressed indices
      _key_norms    (B, H, capacity, 1)          fp16   — per-vector norms
      _key_fp16     (B, H, capacity, D)          fp16   — decompressed working buf

    The fp16 working buffer grows O(1) per decode step (one token appended).
    Call free_working_memory() after generation to release it.
    """

    def __init__(self, bits: int = 4, dtype: torch.dtype = torch.float16,
                 extra_capacity: int = 512):
        """
        Args:
            bits:             Quantization bit-width (2, 3, 4, 8).
            dtype:            Working dtype (fp16 or bf16).
            extra_capacity:   Tokens of headroom to pre-allocate beyond the
                              first prefill length.  Set ≥ n_decode_tokens to
                              avoid any reallocation during generation.
        """
        super().__init__()
        self.bits = bits
        self.dtype = dtype
        self._extra_capacity = extra_capacity

        # Lazy-initialized quantizers keyed by head_dim
        self._quantizers: Dict[int, TurboQuantMSE] = {}

        # Compressed storage (persists after free_working_memory)
        self._key_packed: Dict[int, torch.Tensor] = {}
        self._key_norms:  Dict[int, torch.Tensor] = {}
        self._val_packed: Dict[int, torch.Tensor] = {}
        self._val_norms:  Dict[int, torch.Tensor] = {}

        # Decompressed fp16 working buffers (freed after generation)
        self._key_fp16: Dict[int, torch.Tensor] = {}
        self._val_fp16: Dict[int, torch.Tensor] = {}

        # Per-layer metadata
        self._seq_len: Dict[int, int] = {}
        self._head_dim: Dict[int, int] = {}

        self.stats = QuantizationStats()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_quantizer(self, head_dim: int, device: torch.device) -> TurboQuantMSE:
        if head_dim not in self._quantizers:
            self._quantizers[head_dim] = TurboQuantMSE(
                dim=head_dim, bits=self.bits, device=device, dtype=self.dtype
            )
        return self._quantizers[head_dim]

    def _alloc(self, layer_idx: int, B: int, H: int,
               capacity: int, packed_D: int, head_dim: int,
               device: torch.device):
        """Allocate pre-sized buffers for a new layer."""
        self._key_packed[layer_idx] = torch.empty(B, H, capacity, packed_D, dtype=torch.uint8, device=device)
        self._key_norms [layer_idx] = torch.empty(B, H, capacity, 1,       dtype=self.dtype,  device=device)
        self._val_packed[layer_idx] = torch.empty(B, H, capacity, packed_D, dtype=torch.uint8, device=device)
        self._val_norms [layer_idx] = torch.empty(B, H, capacity, 1,       dtype=self.dtype,  device=device)
        self._key_fp16  [layer_idx] = torch.empty(B, H, capacity, head_dim, dtype=self.dtype,  device=device)
        self._val_fp16  [layer_idx] = torch.empty(B, H, capacity, head_dim, dtype=self.dtype,  device=device)
        self._seq_len   [layer_idx] = 0
        self._head_dim  [layer_idx] = head_dim

    def _grow(self, layer_idx: int):
        """Double buffer capacity (called only when extra_capacity is exceeded)."""
        old_cap = self._key_packed[layer_idx].shape[2]
        new_cap = old_cap * 2
        sl = self._seq_len[layer_idx]
        for buf_dict in (self._key_packed, self._key_norms,
                         self._val_packed, self._val_norms,
                         self._key_fp16,   self._val_fp16):
            if layer_idx not in buf_dict:
                continue
            old = buf_dict[layer_idx]
            new_buf = torch.empty(
                old.shape[0], old.shape[1], new_cap, old.shape[3],
                dtype=old.dtype, device=old.device,
            )
            new_buf[:, :, :sl] = old[:, :, :sl]
            buf_dict[layer_idx] = new_buf

    # ------------------------------------------------------------------
    # DynamicCache interface
    # ------------------------------------------------------------------

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress and store new KV tokens; return full fp16 K and V for attention.

        - L=N on the prefill call (all prompt tokens at once).
        - L=1 on each decode step — this path is O(1) thanks to the
          pre-allocated buffer and per-token decompress.
        """
        B, H, L, D = key_states.shape
        quantizer = self._get_quantizer(D, key_states.device)

        # --- compress new tokens ---
        t0 = time.perf_counter()
        k_packed, k_norms = quantizer.quantize(key_states)
        v_packed, v_norms = quantizer.quantize(value_states)
        self.stats.total_compress_time_ms += (time.perf_counter() - t0) * 1000
        self.stats.total_tokens_compressed += B * H * L
        self.stats.original_bytes   += B * H * L * D * 2
        self.stats.compressed_bytes += (
            k_packed.element_size() * k_packed.numel()
            + k_norms.element_size()  * k_norms.numel()
            + v_packed.element_size() * v_packed.numel()
            + v_norms.element_size()  * v_norms.numel()
        )

        packed_D = k_packed.shape[-1]

        # --- allocate or grow buffer ---
        if layer_idx not in self._seq_len:
            self._alloc(layer_idx, B, H, L + self._extra_capacity,
                        packed_D, D, key_states.device)
        elif self._seq_len[layer_idx] + L > self._key_packed[layer_idx].shape[2]:
            self._grow(layer_idx)

        sl = self._seq_len[layer_idx]

        # --- write compressed in-place ---
        self._key_packed[layer_idx][:, :, sl:sl+L] = k_packed
        self._key_norms [layer_idx][:, :, sl:sl+L] = k_norms
        self._val_packed[layer_idx][:, :, sl:sl+L] = v_packed
        self._val_norms [layer_idx][:, :, sl:sl+L] = v_norms

        # --- decompress only the NEW tokens (O(L), L=1 during decode) ---
        t0 = time.perf_counter()
        k_fp16 = quantizer.dequantize(k_packed, k_norms)
        v_fp16 = quantizer.dequantize(v_packed, v_norms)
        self.stats.total_decompress_time_ms += (time.perf_counter() - t0) * 1000

        # --- write decompressed in-place and advance pointer ---
        self._key_fp16[layer_idx][:, :, sl:sl+L] = k_fp16
        self._val_fp16[layer_idx][:, :, sl:sl+L] = v_fp16
        self._seq_len[layer_idx] = sl + L

        new_sl = sl + L

        # --- return view of full sequence — no allocation, no copy ---
        return (
            self._key_fp16[layer_idx][:, :, :new_sl],
            self._val_fp16[layer_idx][:, :, :new_sl],
        )

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._seq_len.get(layer_idx, 0)

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------

    def free_working_memory(self):
        """
        Release the fp16 working buffers after generation.

        After this call only the compressed representation remains in VRAM,
        giving the actual memory footprint of the compressed KV cache.
        Call this before measuring end-of-generation VRAM.
        """
        self._key_fp16.clear()
        self._val_fp16.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def vram_usage_bytes(self) -> int:
        """Size of compressed storage only (filled portion, not the allocated capacity)."""
        total = 0
        for layer_idx in self._key_packed:
            sl = self._seq_len.get(layer_idx, 0)
            for buf in (self._key_packed[layer_idx], self._key_norms[layer_idx],
                        self._val_packed[layer_idx], self._val_norms[layer_idx]):
                B, H, _, D = buf.shape
                total += B * H * sl * D * buf.element_size()
        return total


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def apply_turboquant_to_model(
    model: PreTrainedModel,
    bits: int = 4,
    n_decode_tokens: int = 512,
    verbose: bool = True,
) -> TurboQuantCache:
    dtype = next(model.parameters()).dtype
    cache = TurboQuantCache(bits=bits, dtype=dtype, extra_capacity=n_decode_tokens)
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
        cache = apply_turboquant_to_model(
            model, bits=bits,
            n_decode_tokens=max_new_tokens,
            verbose=False,
        )
        generate_args["past_key_values"] = cache

    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(**generate_args)
    total_ms = (time.perf_counter() - t0) * 1000

    # Free fp16 working buffer — only compressed storage remains.
    # This must happen before the caller measures end-of-generation VRAM.
    if cache is not None:
        cache.free_working_memory()

    n_new = output_ids.shape[1] - input_ids.shape[1]
    timing = {
        "total_ms": round(total_ms, 1),
        "tokens_per_sec": round(n_new / (total_ms / 1000), 1),
        "n_new_tokens": n_new,
    }
    return output_ids, cache, timing
