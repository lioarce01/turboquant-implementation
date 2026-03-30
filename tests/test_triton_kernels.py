"""
Tests and benchmarks for the TurboQuant Triton kernels.

Correctness tests verify that the Triton kernels produce bitwise-identical
results to the pure PyTorch reference path.

The speed benchmark is informational: it measures the speedup from fusing
argmin+pack and unpack+lookup, and reports the VRAM saved by eliminating the
intermediate (..., D, n_levels) tensor.

Run:
    python tests/test_triton_kernels.py

Requirements:
    pip install triton-windows  (Windows)  or  triton  (Linux)
"""

import sys
import os
import math
import time

import torch
import torch.nn.functional as F
import triton

# Allow running from project root or tests/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.turboquant import (
    TurboQuantMSE,
    pack_indices,
    unpack_indices,
    get_centroids,
)

try:
    from src.triton_kernels import (
        turboquant_quantize_and_pack,
        turboquant_unpack_and_lookup,
    )
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_x_scaled(N: int, D: int, device: torch.device) -> torch.Tensor:
    """Generate random x_scaled that looks like RHT output: ~N(0,1)."""
    return torch.randn(N, D, dtype=torch.float16, device=device)


def _ref_quantize(x_scaled: torch.Tensor, centroids: torch.Tensor, bits: int) -> torch.Tensor:
    """Reference Python path: argmin over expanded tensor + pack."""
    diff = x_scaled.unsqueeze(-1) - centroids       # (..., D, n_levels)
    indices = diff.abs().argmin(dim=-1)              # (..., D) int64
    return pack_indices(indices, bits)               # (..., packed_D) uint8


def _ref_dequantize(packed: torch.Tensor, centroids: torch.Tensor, bits: int, D: int) -> torch.Tensor:
    """Reference Python path: unpack indices + gather centroids."""
    indices = unpack_indices(packed, bits, D)        # (..., D) int16
    return centroids[indices.long()]                 # (..., D) float16


# ---------------------------------------------------------------------------
# Test 1: quantize kernel correctness
# ---------------------------------------------------------------------------

def test_quantize_kernel_correctness():
    """Triton quantize+pack must match the Python reference exactly."""
    if not TRITON_AVAILABLE:
        print("[SKIP] test_quantize_kernel_correctness — Triton not available")
        return

    device = torch.device("cuda")
    bits_list = [2, 4, 8]  # 3-bit intentionally excluded (not supported by kernel)
    shapes = [(1, 128), (16, 128), (256, 128), (1024, 256)]

    all_pass = True
    for bits in bits_list:
        centroids = get_centroids(bits, device, torch.float16)
        for N, D in shapes:
            x_scaled = _make_x_scaled(N, D, device)
            ref = _ref_quantize(x_scaled, centroids, bits)
            got = turboquant_quantize_and_pack(x_scaled, centroids, bits)
            if not torch.equal(ref, got):
                print(f"  [FAIL] bits={bits} shape=({N},{D}): mismatch")
                print(f"    ref[:4]:  {ref.flatten()[:8]}")
                print(f"    got[:4]:  {got.flatten()[:8]}")
                all_pass = False
            else:
                pass  # silent pass

    if all_pass:
        print("[PASS] test_quantize_kernel_correctness")
    return all_pass


# ---------------------------------------------------------------------------
# Test 2: dequantize kernel correctness
# ---------------------------------------------------------------------------

def test_dequantize_kernel_correctness():
    """Triton unpack+lookup must match the Python reference exactly."""
    if not TRITON_AVAILABLE:
        print("[SKIP] test_dequantize_kernel_correctness — Triton not available")
        return

    device = torch.device("cuda")
    bits_list = [2, 4, 8]
    shapes = [(1, 128), (16, 128), (256, 128), (1024, 256)]

    all_pass = True
    for bits in bits_list:
        centroids = get_centroids(bits, device, torch.float16)
        n_per_byte = 8 // bits
        for N, D in shapes:
            # Create valid packed data by using the reference quantizer
            x_scaled = _make_x_scaled(N, D, device)
            packed = _ref_quantize(x_scaled, centroids, bits)

            ref = _ref_dequantize(packed, centroids, bits, D)
            got = turboquant_unpack_and_lookup(packed, centroids, bits, D)

            if not torch.equal(ref, got):
                print(f"  [FAIL] dequantize bits={bits} shape=({N},{D}): mismatch")
                print(f"    ref[:8]:  {ref.flatten()[:8]}")
                print(f"    got[:8]:  {got.flatten()[:8]}")
                all_pass = False

    if all_pass:
        print("[PASS] test_dequantize_kernel_correctness")
    return all_pass


# ---------------------------------------------------------------------------
# Test 3: end-to-end round-trip via TurboQuantMSE
# ---------------------------------------------------------------------------

def test_end_to_end_roundtrip():
    """
    TurboQuantMSE with Triton kernels active must give the same MSE as the
    Python path (i.e. the quantization quality is unchanged).
    """
    if not TRITON_AVAILABLE:
        print("[SKIP] test_end_to_end_roundtrip — Triton not available")
        return

    device = torch.device("cuda")
    dim = 128
    bits = 4

    torch.manual_seed(0)
    x = torch.randn(2, 8, 64, dim, dtype=torch.float16, device=device)

    # Triton path (active because _TRITON_AVAILABLE == True in turboquant.py)
    tq = TurboQuantMSE(dim=dim, bits=bits, device=device)
    packed_triton, norms_triton = tq.quantize(x)
    x_hat_triton = tq.dequantize(packed_triton, norms_triton)
    mse_triton = ((x - x_hat_triton) ** 2).mean().item()

    # Force Python path by temporarily patching
    import src.turboquant as tq_module
    original = tq_module._TRITON_AVAILABLE
    tq_module._TRITON_AVAILABLE = False
    try:
        tq2 = TurboQuantMSE(dim=dim, bits=bits, device=device)
        packed_py, norms_py = tq2.quantize(x)
        x_hat_py = tq2.dequantize(packed_py, norms_py)
        mse_py = ((x - x_hat_py) ** 2).mean().item()
    finally:
        tq_module._TRITON_AVAILABLE = original

    # MSE must be identical (both paths are deterministic, same seed)
    if abs(mse_triton - mse_py) < 1e-7:
        print(f"[PASS] test_end_to_end_roundtrip  (MSE={mse_triton:.6f})")
    else:
        print(f"[FAIL] test_end_to_end_roundtrip")
        print(f"  Triton MSE: {mse_triton:.8f}")
        print(f"  Python MSE: {mse_py:.8f}")
        print(f"  Diff:       {abs(mse_triton - mse_py):.2e}")


# ---------------------------------------------------------------------------
# Benchmark: Triton vs PyTorch — speed and VRAM
# ---------------------------------------------------------------------------

def benchmark_kernels():
    """
    Compare Triton kernel vs Python reference on a realistic prefill shape.
    Reports: latency (ms), speedup, and VRAM saved by eliminating the
    intermediate (..., D, n_levels) tensor.
    """
    if not TRITON_AVAILABLE:
        print("[SKIP] benchmark_kernels — Triton not available")
        return

    device = torch.device("cuda")
    bits = 4
    B, H, L, D = 1, 8, 512, 128   # Llama 3.2-3B typical prefill shape
    N = B * H * L
    n_levels = 2 ** bits
    n_per_byte = 8 // bits

    centroids = get_centroids(bits, device, torch.float16)

    print(f"\n{'='*60}")
    print(f"  Kernel Benchmark — B={B} H={H} L={L} D={D} bits={bits}")
    print(f"{'='*60}")

    # Intermediate tensor size: (N, D, n_levels)
    intermediate_bytes = N * D * n_levels * 2  # float16
    print(f"  Intermediate tensor size (eliminated by kernel): "
          f"{intermediate_bytes / 1e6:.1f} MB")

    n_warmup = 20
    n_bench  = 200

    # --- Python reference: quantize ---
    x = _make_x_scaled(N, D, device)

    for _ in range(n_warmup):
        _ = _ref_quantize(x, centroids, bits)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_bench):
        _ = _ref_quantize(x, centroids, bits)
    torch.cuda.synchronize()
    py_q_ms = (time.perf_counter() - t0) * 1000 / n_bench

    # --- Triton kernel: quantize ---
    for _ in range(n_warmup):
        _ = turboquant_quantize_and_pack(x, centroids, bits)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_bench):
        _ = turboquant_quantize_and_pack(x, centroids, bits)
    torch.cuda.synchronize()
    triton_q_ms = (time.perf_counter() - t0) * 1000 / n_bench

    print(f"\n  Quantize (argmin + pack):")
    print(f"    Python path:    {py_q_ms:.4f} ms")
    print(f"    Triton kernel:  {triton_q_ms:.4f} ms")
    if triton_q_ms > 0:
        print(f"    Speedup:        {py_q_ms / triton_q_ms:.2f}x")

    # --- Python reference: dequantize ---
    packed = _ref_quantize(x, centroids, bits)

    for _ in range(n_warmup):
        _ = _ref_dequantize(packed, centroids, bits, D)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_bench):
        _ = _ref_dequantize(packed, centroids, bits, D)
    torch.cuda.synchronize()
    py_dq_ms = (time.perf_counter() - t0) * 1000 / n_bench

    # --- Triton kernel: dequantize ---
    for _ in range(n_warmup):
        _ = turboquant_unpack_and_lookup(packed, centroids, bits, D)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_bench):
        _ = turboquant_unpack_and_lookup(packed, centroids, bits, D)
    torch.cuda.synchronize()
    triton_dq_ms = (time.perf_counter() - t0) * 1000 / n_bench

    print(f"\n  Dequantize (unpack + lookup):")
    print(f"    Python path:    {py_dq_ms:.4f} ms")
    print(f"    Triton kernel:  {triton_dq_ms:.4f} ms")
    if triton_dq_ms > 0:
        print(f"    Speedup:        {py_dq_ms / triton_dq_ms:.2f}x")

    print(f"\n  Combined quantize+dequantize:")
    print(f"    Python:  {py_q_ms + py_dq_ms:.4f} ms")
    print(f"    Triton:  {triton_q_ms + triton_dq_ms:.4f} ms")
    if (triton_q_ms + triton_dq_ms) > 0:
        print(f"    Speedup: {(py_q_ms + py_dq_ms) / (triton_q_ms + triton_dq_ms):.2f}x")
    print(f"{'='*60}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("[ERROR] CUDA not available — Triton kernels require a GPU.")
        sys.exit(1)

    if not TRITON_AVAILABLE:
        print("[ERROR] Triton not installed.")
        print("  Install with: pip install triton-windows")
        sys.exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Triton version: {triton.__version__}")
    print()

    # Correctness tests
    test_quantize_kernel_correctness()
    test_dequantize_kernel_correctness()
    test_end_to_end_roundtrip()

    # Speed benchmark
    benchmark_kernels()
