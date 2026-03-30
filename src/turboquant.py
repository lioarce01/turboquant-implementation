"""
TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate
Implementation of algorithms from https://arxiv.org/abs/2504.19874

Core components:
  - Randomized Hadamard Transform (RHT) for rotation
  - Lloyd-Max scalar quantizer for Gaussian distributions
  - TurboQuant_MSE: minimizes mean squared error
  - TurboQuant_PROD: minimizes inner product distortion (2-stage: MSE + QJL)
"""

import math
import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import brentq
from typing import Tuple, Optional

# Optional Triton kernels — used when available (CUDA only, bits in {1,2,4,8}).
# Falls back to pure PyTorch for CPU, 3-bit, or when triton-windows is not installed.
try:
    from .triton_kernels import turboquant_quantize_and_pack, turboquant_unpack_and_lookup
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


# ---------------------------------------------------------------------------
# Index packing / unpacking
# Stores centroid indices in compact uint8 format instead of int16,
# giving real memory savings that match the quantization bit-width.
#
# Compression vs fp16 (2 bytes per element):
#   4-bit → nibble pack (2/byte)  → 4x compression
#   2-bit → quad pack  (4/byte)  → 8x compression
#   8-bit → uint8      (1/byte)  → 2x compression
#   3-bit → uint8      (1/byte)  → 2x compression  (packing 3-bit is non-trivial)
# ---------------------------------------------------------------------------

def pack_indices(indices: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack integer indices into compact uint8 storage."""
    if bits == 8 or bits == 3:
        # 1 index per byte — uint8 still halves storage vs int16
        return indices.to(torch.uint8)

    n_per_byte = 8 // bits          # 2 for 4-bit, 4 for 2-bit, 8 for 1-bit
    D = indices.shape[-1]
    pad = (-D) % n_per_byte
    if pad:
        indices = F.pad(indices, (0, pad), value=0)

    mask = (1 << bits) - 1
    packed = torch.zeros(*indices.shape[:-1], (D + pad) // n_per_byte,
                         dtype=torch.uint8, device=indices.device)
    for i in range(n_per_byte):
        packed |= (indices[..., i::n_per_byte].to(torch.uint8) & mask) << (i * bits)
    return packed


def unpack_indices(packed: torch.Tensor, bits: int, original_dim: int) -> torch.Tensor:
    """Unpack uint8 bytes back to int16 indices."""
    if bits == 8 or bits == 3:
        return packed.to(torch.int16)[..., :original_dim]

    n_per_byte = 8 // bits
    mask = int((1 << bits) - 1)
    D_padded = packed.shape[-1] * n_per_byte
    result = torch.zeros(*packed.shape[:-1], D_padded,
                         dtype=torch.int16, device=packed.device)
    for i in range(n_per_byte):
        result[..., i::n_per_byte] = (packed >> (i * bits)).to(torch.int16) & mask
    return result[..., :original_dim]


# ---------------------------------------------------------------------------
# Precomputed Lloyd-Max centroids for Gaussian N(0,1)
# These are the theoretically optimal centroids for quantizing Gaussian data.
# The scaled version for N(0, 1/d) is centroids / sqrt(d).
# Source: solved via the Lloyd-Max algorithm offline.
# ---------------------------------------------------------------------------

def _compute_lloyd_max_centroids(n_levels: int, n_iter: int = 1000) -> np.ndarray:
    """
    Compute Lloyd-Max optimal centroids for Gaussian N(0,1).
    Uses the fixed-point iteration: centroid = E[X | X in Voronoi cell].
    """
    from scipy.stats import norm

    # Initialize: evenly spaced quantiles
    probs = np.linspace(0.5 / n_levels, 1 - 0.5 / n_levels, n_levels)
    centroids = norm.ppf(probs)

    for _ in range(n_iter):
        # Boundaries: midpoints between consecutive centroids
        boundaries = np.concatenate([[-np.inf], (centroids[:-1] + centroids[1:]) / 2, [np.inf]])

        # Update each centroid: conditional mean of Gaussian given cell
        new_centroids = np.empty(n_levels)
        for i in range(n_levels):
            lo, hi = boundaries[i], boundaries[i + 1]
            # E[X | lo < X < hi] for X ~ N(0,1)
            p_lo = norm.pdf(lo) if not np.isinf(lo) else 0.0
            p_hi = norm.pdf(hi) if not np.isinf(hi) else 0.0
            denom = norm.cdf(hi) - norm.cdf(lo)
            if denom < 1e-12:
                new_centroids[i] = centroids[i]
            else:
                new_centroids[i] = (p_lo - p_hi) / denom

        if np.max(np.abs(new_centroids - centroids)) < 1e-10:
            break
        centroids = new_centroids

    return centroids


# Cache of precomputed centroids: {n_bits: tensor of shape (2**n_bits,)}
_CENTROID_CACHE: dict[int, torch.Tensor] = {}


def get_centroids(n_bits: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return Lloyd-Max centroids for Gaussian N(0,1) at given bit-width."""
    if n_bits not in _CENTROID_CACHE:
        n_levels = 2 ** n_bits
        centroids = _compute_lloyd_max_centroids(n_levels)
        _CENTROID_CACHE[n_bits] = torch.tensor(centroids, dtype=torch.float32)
    return _CENTROID_CACHE[n_bits].to(device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Randomized Hadamard Transform (RHT)
# Fast O(d log d) rotation that spreads out any input vector.
# ---------------------------------------------------------------------------

def _next_power_of_2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def _fwht_iterative(x: torch.Tensor) -> torch.Tensor:
    """
    Fast Walsh-Hadamard Transform in-place on last dimension.
    Input must have last dim as a power of 2.
    """
    n = x.shape[-1]
    h = 1
    while h < n:
        x = x.clone()
        x[..., :n:2*h], x[..., h:n:2*h] = (
            x[..., :n:2*h] + x[..., h:n:2*h],
            x[..., :n:2*h] - x[..., h:n:2*h],
        )
        # The slice assignment above doesn't work cleanly for non-contiguous;
        # use a reshape-based approach instead.
        h *= 2
    return x


def fwht(x: torch.Tensor) -> torch.Tensor:
    """
    Fast Walsh-Hadamard Transform along the last dimension.
    Expects last dim to be a power of 2.
    """
    n = x.shape[-1]
    assert (n & (n - 1)) == 0, f"Last dim must be power of 2, got {n}"
    h = 1
    out = x.clone()
    while h < n:
        out = out.reshape(*out.shape[:-1], n // (2 * h), 2 * h)
        left = out[..., :h].clone()
        right = out[..., h:].clone()
        out[..., :h] = left + right
        out[..., h:] = left - right
        out = out.reshape(*out.shape[:-2], n)
        h *= 2
    return out


class RandomizedHadamardTransform:
    """
    Randomized Hadamard Transform: apply random ±1 signs then WHT.
    Produces a rotation with the same statistical properties as a
    full random Gaussian rotation but in O(d log d).
    """

    def __init__(self, dim: int, device: torch.device, dtype: torch.dtype, seed: int = 42):
        self.dim = dim
        self.padded_dim = _next_power_of_2(dim)
        self.device = device
        self.dtype = dtype

        # Fixed random sign vector (shared across calls for consistency)
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)
        self.signs = (
            torch.randint(0, 2, (self.padded_dim,), generator=rng, device=device) * 2 - 1
        ).to(dtype)
        self.scale = 1.0 / math.sqrt(self.padded_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RHT. Input: (..., dim). Output: (..., padded_dim)."""
        orig_shape = x.shape
        if self.padded_dim > self.dim:
            pad = self.padded_dim - self.dim
            x = F.pad(x, (0, pad))

        x = x * self.signs
        x = fwht(x)
        x = x * self.scale
        return x

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse RHT. Input: (..., padded_dim). Output: (..., dim)."""
        # WHT is its own inverse up to a scale factor
        x = x / self.scale
        x = fwht(x)
        x = x * self.signs
        x = x / self.padded_dim  # WHT normalization
        return x[..., : self.dim]


# ---------------------------------------------------------------------------
# TurboQuant MSE
# Quantizes vectors minimizing mean squared error.
# ---------------------------------------------------------------------------

class TurboQuantMSE:
    """
    TurboQuant_MSE: Quantize vectors minimizing MSE.

    Usage:
        tq = TurboQuantMSE(dim=128, bits=4, device=device)
        indices = tq.quantize(keys)      # compress: (B, H, L, D) -> (B, H, L, D) int tensor
        keys_hat = tq.dequantize(indices) # decompress: back to (B, H, L, D) float
    """

    def __init__(self, dim: int, bits: int, device: torch.device, dtype: torch.dtype = torch.float16, seed: int = 42):
        assert 1 <= bits <= 8, f"bits must be in [1, 8], got {bits}"
        self.dim = dim
        self.bits = bits
        self.device = device
        self.dtype = dtype

        self.rht = RandomizedHadamardTransform(dim, device, dtype, seed)
        self.centroids = get_centroids(bits, device, dtype)  # shape: (2**bits,)

    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize vectors.
        Args:
            x: float tensor of shape (..., dim)
        Returns:
            packed:  uint8 tensor, indices packed for storage (4-bit → 2/byte, etc.)
            norms:   float tensor (..., 1)
        """
        x = x.to(self.dtype)

        norms = x.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        x_normalized = x / norms

        x_rot = self.rht.forward(x_normalized)                       # (..., padded_dim)
        scale = math.sqrt(self.rht.padded_dim)
        x_scaled = x_rot * scale

        if _TRITON_AVAILABLE and x_scaled.is_cuda and self.bits != 3:
            # Fused argmin + pack — no intermediate (..., D, n_levels) tensor
            packed = turboquant_quantize_and_pack(x_scaled, self.centroids, self.bits)
        else:
            # Pure PyTorch fallback (CPU, 3-bit, or Triton unavailable)
            diff = x_scaled.unsqueeze(-1) - self.centroids           # (..., padded_dim, n_levels)
            indices = diff.abs().argmin(dim=-1)                      # (..., padded_dim) int64
            packed = pack_indices(indices, self.bits)                 # (..., packed_dim) uint8
        return packed, norms

    def dequantize(self, packed: torch.Tensor, norms: torch.Tensor) -> torch.Tensor:
        """
        Dequantize compressed vectors.
        Args:
            packed: uint8 tensor from quantize()
            norms:  float tensor (..., 1)
        Returns:
            x_hat: float tensor (..., dim)
        """
        if _TRITON_AVAILABLE and packed.is_cuda and self.bits != 3:
            # Fused unpack + centroid lookup — no temporary int16 indices tensor
            x_scaled = turboquant_unpack_and_lookup(
                packed, self.centroids, self.bits, self.rht.padded_dim
            )
        else:
            # Pure PyTorch fallback
            indices = unpack_indices(packed, self.bits, self.rht.padded_dim)  # (..., padded_dim)
            x_scaled = self.centroids[indices.long()]  # (..., padded_dim)

        # Undo scaling
        scale = math.sqrt(self.rht.padded_dim)
        x_rot = x_scaled / scale

        # Inverse rotation
        x_normalized = self.rht.inverse(x_rot)  # (..., dim)

        # Restore original norms
        x_hat = x_normalized * norms
        return x_hat.to(self.dtype)

    def compress_ratio(self) -> float:
        """Compression ratio relative to float16."""
        return 16.0 / self.bits


# ---------------------------------------------------------------------------
# Quantized Johnson-Lindenstrauss (QJL) for residual inner products
# ---------------------------------------------------------------------------

class QJLSketch:
    """
    1-bit Quantized Johnson-Lindenstrauss sketch for inner product estimation.
    Given residual r, stores sign(Φr) where Φ has entries ±1/sqrt(m).

    Unbiased estimator for <q, r> (q full-precision, r sketched):
        Derivation: E[(Φq)_i · sign(Φr)_i] = <q,r>·√(2/π) / (||r||·√m)
        Summed over m:  E[<Φq, sign(Φr)>] = √m · <q,r> · √(2/π) / ||r||
        Solving:        <q,r> ≈ ||r|| · √(π/2) / √m · <Φq, sign(Φr)>
    """

    def __init__(self, dim: int, sketch_dim: int, device: torch.device, dtype: torch.dtype, seed: int = 99):
        self.dim = dim
        self.sketch_dim = sketch_dim
        self.device = device
        self.dtype = dtype

        rng = torch.Generator(device=device)
        rng.manual_seed(seed)
        # Random projection matrix Φ: shape (sketch_dim, dim), entries ±1/sqrt(sketch_dim)
        self.phi = (
            torch.randint(0, 2, (sketch_dim, dim), generator=rng, device=device).float() * 2 - 1
        ).to(dtype) / math.sqrt(sketch_dim)

    def sketch(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute 1-bit sketch.
        Args:
            x: (..., dim)
        Returns:
            bits: (..., sketch_dim) dtype=torch.int8, values in {-1, +1}
        """
        # Project: (..., sketch_dim)
        projected = x @ self.phi.T
        return projected.sign().to(torch.int8)

    def estimate_inner_product(
        self,
        q: torch.Tensor,
        r_sketch: torch.Tensor,
        r_norms: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate inner product <q, r> given 1-bit sketch of residual r.
        Args:
            q: full-precision query (..., dim)
            r_sketch: sign(Φr), int8 (..., sketch_dim), values ±1
            r_norms: ||r|| per vector (..., 1)
        Returns:
            ip_estimate: (...,) unbiased estimate of <q, r>
        """
        # Project query through same Φ (continuous)
        q_proj = q @ self.phi.T  # (..., sketch_dim)  — Φ entries ±1/√m

        # Dot product: <Φq, sign(Φr)>
        raw = (q_proj * r_sketch.to(self.dtype)).sum(dim=-1)  # (...,)

        # Unbiased scale: ||r|| * √(π/2) / √m
        # Derived from E[<Φq, sign(Φr)>] = √m · <q,r> · √(2/π) / ||r||
        scale = r_norms.squeeze(-1) * math.sqrt(math.pi / 2) / math.sqrt(self.sketch_dim)
        return raw * scale


# ---------------------------------------------------------------------------
# TurboQuant PROD
# Two-stage: MSE quantize with (b-1) bits + QJL on residual for unbiased IP
# ---------------------------------------------------------------------------

class TurboQuantPROD:
    """
    TurboQuant_PROD: Quantize vectors for inner product estimation.

    Stage 1: Apply TurboQuantMSE with (bits-1) bits
    Stage 2: Compute residual, apply 1-bit QJL sketch

    This provides an unbiased estimator of inner products between queries and keys.
    """

    def __init__(
        self,
        dim: int,
        bits: int,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
        qjl_sketch_dim: Optional[int] = None,
        seed: int = 42,
    ):
        assert bits >= 2, f"PROD mode requires bits >= 2 (1 for MSE + 1 for QJL), got {bits}"
        self.dim = dim
        self.bits = bits
        self.device = device
        self.dtype = dtype

        # Stage 1: MSE quantizer with (bits-1) bits
        self.mse_quantizer = TurboQuantMSE(dim, bits - 1, device, dtype, seed)

        # Stage 2: QJL sketch using 1 bit across sketch_dim dims
        # sketch_dim can match dim for best quality; or use dim//2 to save memory
        if qjl_sketch_dim is None:
            qjl_sketch_dim = dim  # 1 bit per original dimension
        self.qjl = QJLSketch(dim, qjl_sketch_dim, device, dtype, seed + 1)

    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize for inner product estimation.
        Returns:
            mse_indices: int16 (..., padded_dim)
            mse_norms: float (..., 1)
            residual_sketch: int8 (..., qjl_sketch_dim)
            residual_norms: float (..., 1)  — needed for unbiased QJL estimator
        """
        # Stage 1: MSE quantize
        mse_indices, mse_norms = self.mse_quantizer.quantize(x)

        # Compute residual r = x - x̂
        x_approx = self.mse_quantizer.dequantize(mse_indices, mse_norms)
        residual = x.to(self.dtype) - x_approx

        # Store residual norm (required for unbiased QJL scaling)
        residual_norms = residual.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        # Stage 2: QJL sketch of residual
        residual_sketch = self.qjl.sketch(residual)

        return mse_indices, mse_norms, residual_sketch, residual_norms

    def estimate_inner_product(
        self,
        q: torch.Tensor,
        k_mse_indices: torch.Tensor,
        k_mse_norms: torch.Tensor,
        k_residual_sketch: torch.Tensor,
        k_residual_norms: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate <q, k> from compressed k representation.
        Args:
            q: query (..., dim)
            k_mse_indices, k_mse_norms: MSE-compressed key
            k_residual_sketch: 1-bit QJL sketch of key residual
            k_residual_norms: norms of the residual (..., 1)
        Returns:
            ip_estimate: estimated inner product
        """
        # Stage 1: <q, k̂> via MSE reconstruction
        k_approx = self.mse_quantizer.dequantize(k_mse_indices, k_mse_norms)
        ip_mse = (q.to(self.dtype) * k_approx).sum(dim=-1)

        # Stage 2: <q, r> via QJL estimator (r = k - k̂)
        ip_residual = self.qjl.estimate_inner_product(
            q.to(self.dtype), k_residual_sketch, k_residual_norms
        )

        return ip_mse + ip_residual

    def compress_ratio(self) -> float:
        """Effective bits per element including QJL."""
        mse_bits = self.bits - 1
        qjl_bits = self.qjl.sketch_dim / self.dim  # 1 bit per sketch dim
        total_bits = mse_bits + qjl_bits
        return 16.0 / total_bits


# ---------------------------------------------------------------------------
# Utility: measure MSE distortion empirically
# ---------------------------------------------------------------------------

def measure_mse_distortion(quantizer: TurboQuantMSE, n_vectors: int = 1000, dim: int = 128) -> float:
    """Measure empirical MSE distortion of the quantizer on random unit vectors."""
    x = torch.randn(n_vectors, dim, device=quantizer.device, dtype=quantizer.dtype)
    norms = x.norm(dim=-1, keepdim=True)
    x = x / norms  # unit norm

    indices, stored_norms = quantizer.quantize(x)
    x_hat = quantizer.dequantize(indices, stored_norms)

    mse = ((x - x_hat) ** 2).mean().item()
    theoretical_bound = (math.sqrt(3) * math.pi / 2) / (4 ** quantizer.bits)
    info_theoretic_lower = 1.0 / (4 ** quantizer.bits)

    print(f"  Empirical MSE:          {mse:.6f}")
    print(f"  Theoretical bound:      {theoretical_bound:.6f}  (paper Theorem 1)")
    print(f"  Info-theoretic lower:   {info_theoretic_lower:.6f}")
    print(f"  Ratio to lower bound:   {mse / info_theoretic_lower:.2f}x  (paper claims ≤2.72x)")
    return mse
