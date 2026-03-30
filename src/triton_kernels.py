"""
Triton GPU kernels for TurboQuant.

Requires: triton-windows  (pip install triton-windows)
Fallback:  src/turboquant.py uses pure Python/PyTorch when Triton is unavailable.

Problem these kernels solve
---------------------------
In TurboQuantMSE.quantize() the naive PyTorch path does:

    diff = x_scaled.unsqueeze(-1) - centroids   # (..., D, n_levels)  ← 64 MB intermediate!
    indices = diff.abs().argmin(dim=-1)          # (..., D)
    packed = pack_indices(indices, bits)         # Python loop, sequential

For a typical prefill (B=1, H=8, L=512, D=128, n_levels=16) this allocates a
64 MB tensor that is immediately thrown away.  The pack loop is also sequential.

Kernel design (Triton 3.x compatible)
--------------------------------------
Both kernels use one Triton program per row (one token×head vector).

Kernel 1 — quantize_and_pack:
  Grid: (N,)  where N = B*H*L
  Per program, BLOCK_BYTES = packed_D bytes are processed.

  For each slot s in 0..n_per_byte-1  (tl.static_range → fully unrolled):
    1. Load BLOCK_BYTES input elements x[j*n_per_byte + s] for j in 0..BLOCK_BYTES-1
    2. For each centroid c in 0..n_levels-1  (tl.static_range → fully unrolled):
         Load centroids[c] as a scalar (one load, broadcast to BLOCK_BYTES)
         Compare |x - cv| against running best_dist
         Update best_idx where this centroid is closer
    3. Pack best_idx into the correct bit field of the output byte

  Result: zero global-memory intermediate tensor.
  Centroids (16 × 2B = 32B for 4-bit) are scalar-loaded and live in registers.

Kernel 2 — unpack_and_lookup:
  Grid: (N,)
  Per program, BLOCK_BYTES packed bytes are read and D output elements are written.

  For each slot s in 0..n_per_byte-1  (unrolled):
    1. Extract centroid index from packed byte: idx = (byte >> s*bits) & mask
    2. For each centroid c in 0..n_levels-1  (unrolled):
         result = where(idx == c, centroids[c], result)
    3. Store float16 result at output position j*n_per_byte + s

Key Triton 3.x compatibility notes
------------------------------------
  - BLOCK_BYTES is a separate tl.constexpr parameter (not computed inside kernel
    via constexpr arithmetic, which can produce a tensor instead of an int in
    Triton 3.x and fail in tl.zeros/tl.full shape arguments).
  - Centroids are loaded as scalars with tl.load(ptr + c) inside tl.static_range —
    c is a Python int at trace time, so this is fully unrolled.
  - No tl.gather, no tl.reshape, no tl.argmin on 2D tensors.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel 1: quantize_and_pack
# ---------------------------------------------------------------------------

@triton.jit
def _quantize_and_pack_kernel(
    x_scaled_ptr,    # (N, D) float16 — rotated & scaled vectors
    centroids_ptr,   # (n_levels,) float16 — Lloyd-Max centroids
    out_packed_ptr,  # (N, packed_D) uint8 — output
    N, D, packed_D,
    n_levels:   tl.constexpr,   # 2**bits
    n_per_byte: tl.constexpr,   # 8 // bits
    bits:       tl.constexpr,   # quantization bit-width
    BLOCK_BYTES: tl.constexpr,  # tile width over output bytes (>= packed_D, power of 2)
):
    row = tl.program_id(0)
    if row >= N:
        return

    mask_bits = (1 << bits) - 1

    # Output byte indices for this tile
    byte_offsets = tl.arange(0, BLOCK_BYTES)   # (BLOCK_BYTES,)
    byte_mask    = byte_offsets < packed_D

    # Accumulator for the packed output bytes
    packed_out = tl.zeros([BLOCK_BYTES], dtype=tl.uint8)

    # --- Outer loop: one iteration per slot within a byte ---
    # tl.static_range unrolls at compile time → no runtime branch overhead
    for slot in tl.static_range(n_per_byte):

        # Input positions: for output byte j and slot s, input is j*n_per_byte + s
        src_col  = byte_offsets * n_per_byte + slot
        src_mask = byte_mask & (src_col < D)

        # Load BLOCK_BYTES input elements — keep float16 to match the Python
        # reference path (which does fp16 subtraction).  Using float32 here
        # can break ties differently and cause correctness mismatches.
        x = tl.load(
            x_scaled_ptr + row * D + src_col,
            mask=src_mask,
            other=0.0,
        )                                       # (BLOCK_BYTES,) float16

        # --- Inner loop: linear scan over centroids ---
        # All n_levels iterations are unrolled; each centroid is a scalar register.
        # No intermediate (BLOCK_BYTES, n_levels) tensor is ever written to memory.
        # 100.0 is safely above any |x - centroid| for normalized RHT output.
        best_dist = tl.full([BLOCK_BYTES], value=100.0, dtype=tl.float16)
        best_idx  = tl.zeros([BLOCK_BYTES], dtype=tl.int32)

        for c in tl.static_range(n_levels):
            # Load single centroid value as scalar float16 — stays in registers
            cv = tl.load(centroids_ptr + c)     # scalar float16
            dist   = tl.abs(x - cv)             # (BLOCK_BYTES,) float16, matches Python
            closer = dist < best_dist
            best_dist = tl.where(closer, dist,  best_dist)
            best_idx  = tl.where(closer, c,     best_idx)

        # Pack: shift slot's indices into correct bit position and OR into output
        packed_out = packed_out | (
            (best_idx & mask_bits).to(tl.uint8) << (slot * bits)
        )

    tl.store(out_packed_ptr + row * packed_D + byte_offsets, packed_out, mask=byte_mask)


# ---------------------------------------------------------------------------
# Kernel 2: unpack_and_lookup
# ---------------------------------------------------------------------------

@triton.jit
def _unpack_and_lookup_kernel(
    packed_ptr,    # (N, packed_D) uint8
    centroids_ptr, # (n_levels,) float16
    out_x_ptr,     # (N, D) float16 — output
    N, D, packed_D,
    n_levels:   tl.constexpr,
    n_per_byte: tl.constexpr,
    bits:       tl.constexpr,
    BLOCK_BYTES: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= N:
        return

    mask_bits = (1 << bits) - 1

    byte_offsets = tl.arange(0, BLOCK_BYTES)
    byte_mask    = byte_offsets < packed_D

    # Load all packed bytes for this row in one coalesced load
    packed_bytes = tl.load(
        packed_ptr + row * packed_D + byte_offsets,
        mask=byte_mask,
        other=0,
    ).to(tl.uint8)                              # (BLOCK_BYTES,)

    # --- Outer loop: one iteration per slot within a byte ---
    for slot in tl.static_range(n_per_byte):

        # Extract centroid index for this slot
        idx = ((packed_bytes >> (slot * bits)) & mask_bits).to(tl.int32)  # (BLOCK_BYTES,)

        # --- Inner loop: lookup centroid value via linear scan ---
        # tl.where over all n_levels centroids, fully unrolled.
        result = tl.zeros([BLOCK_BYTES], dtype=tl.float32)
        for c in tl.static_range(n_levels):
            cv     = tl.load(centroids_ptr + c).to(tl.float32)  # scalar
            result = tl.where(idx == c, cv, result)

        # Write float16 output at positions j*n_per_byte + slot
        out_col  = byte_offsets * n_per_byte + slot
        out_mask = byte_mask & (out_col < D)
        tl.store(
            out_x_ptr + row * D + out_col,
            result.to(tl.float16),
            mask=out_mask,
        )


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------

def turboquant_quantize_and_pack(
    x_scaled: torch.Tensor,
    centroids: torch.Tensor,
    bits: int,
) -> torch.Tensor:
    """
    Fused argmin + pack kernel.

    Args:
        x_scaled:  (..., padded_D) float16 — rotated and scaled vectors from RHT
        centroids: (n_levels,) float16 — Lloyd-Max centroids
        bits:      quantization bit-width (1, 2, 4, or 8; 3-bit not supported)

    Returns:
        packed: (..., packed_D) uint8   where packed_D = padded_D // (8 // bits)
    """
    assert bits in (1, 2, 4, 8), (
        f"Triton kernel supports bits in {{1, 2, 4, 8}}, got {bits}. "
        "Use the Python fallback for 3-bit."
    )
    assert x_scaled.dtype == torch.float16, "x_scaled must be float16"
    assert centroids.dtype == torch.float16, "centroids must be float16"
    assert x_scaled.is_cuda, "x_scaled must be on CUDA"

    original_shape = x_scaled.shape
    D        = original_shape[-1]
    n_levels = centroids.shape[0]
    n_per_byte = 8 // bits
    packed_D   = D // n_per_byte

    N      = x_scaled.numel() // D
    x_flat = x_scaled.reshape(N, D).contiguous()
    cents  = centroids.contiguous()

    # BLOCK_BYTES must be a power of 2 and >= packed_D.
    # For typical head_dim=128 and 4-bit: packed_D=64, BLOCK_BYTES=64.
    BLOCK_BYTES = max(64, triton.next_power_of_2(packed_D))

    packed_out = torch.empty(N, packed_D, dtype=torch.uint8, device=x_scaled.device)

    _quantize_and_pack_kernel[(N,)](
        x_flat, cents, packed_out,
        N, D, packed_D,
        n_levels=n_levels,
        n_per_byte=n_per_byte,
        bits=bits,
        BLOCK_BYTES=BLOCK_BYTES,
    )

    return packed_out.reshape(*original_shape[:-1], packed_D)


def turboquant_unpack_and_lookup(
    packed: torch.Tensor,
    centroids: torch.Tensor,
    bits: int,
    original_dim: int,
) -> torch.Tensor:
    """
    Fused unpack + centroid-lookup kernel.

    Args:
        packed:       (..., packed_D) uint8
        centroids:    (n_levels,) float16
        bits:         quantization bit-width (must match what was used to pack)
        original_dim: padded_dim (= power-of-2 head_dim after RHT padding)

    Returns:
        x_scaled: (..., original_dim) float16
    """
    assert bits in (1, 2, 4, 8), (
        f"Triton kernel supports bits in {{1, 2, 4, 8}}, got {bits}. "
        "Use the Python fallback for 3-bit."
    )
    assert packed.dtype == torch.uint8, "packed must be uint8"
    assert centroids.dtype == torch.float16, "centroids must be float16"
    assert packed.is_cuda, "packed must be on CUDA"

    original_shape = packed.shape
    packed_D   = original_shape[-1]
    n_levels   = centroids.shape[0]
    n_per_byte = 8 // bits
    D          = original_dim

    N           = packed.numel() // packed_D
    packed_flat = packed.reshape(N, packed_D).contiguous()
    cents       = centroids.contiguous()

    BLOCK_BYTES = max(64, triton.next_power_of_2(packed_D))

    x_out = torch.empty(N, D, dtype=torch.float16, device=packed.device)

    _unpack_and_lookup_kernel[(N,)](
        packed_flat, cents, x_out,
        N, D, packed_D,
        n_levels=n_levels,
        n_per_byte=n_per_byte,
        bits=bits,
        BLOCK_BYTES=BLOCK_BYTES,
    )

    return x_out.reshape(*original_shape[:-1], D)
