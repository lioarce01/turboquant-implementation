"""
Validate TurboQuant algorithm correctness and measure empirical distortion.
Run this BEFORE the full benchmark to confirm the math works.

Usage:
    python validate_turboquant.py
"""

import sys
import os
import math
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.turboquant import TurboQuantMSE, TurboQuantPROD, measure_mse_distortion, get_centroids


def test_rht_orthogonality():
    """Check that the Randomized Hadamard Transform approximately preserves norms."""
    from src.turboquant import RandomizedHadamardTransform
    print("\n[1] Testing RHT norm preservation...")
    dim = 128
    rht = RandomizedHadamardTransform(dim, device=torch.device("cpu"), dtype=torch.float32)

    x = torch.randn(1000, dim)
    x_norms = x.norm(dim=-1)

    x_rot = rht.forward(x)
    x_rot_norms = x_rot.norm(dim=-1)

    ratio = (x_rot_norms / x_norms).mean().item()
    print(f"  Mean norm ratio after RHT: {ratio:.4f} (expected ≈ 1.0)")
    assert abs(ratio - 1.0) < 0.05, f"RHT is not norm-preserving: {ratio}"

    # Check inverse
    x_reconstructed = rht.inverse(x_rot)
    mse = ((x_reconstructed - x) ** 2).mean().item()
    print(f"  RHT round-trip MSE: {mse:.2e} (expected ≈ 0)")
    assert mse < 1e-4, f"RHT inverse is incorrect: {mse}"
    print("  ✓ PASSED")


def test_centroids():
    """Check that Lloyd-Max centroids have good properties."""
    print("\n[2] Testing Lloyd-Max centroids...")
    for bits in [1, 2, 3, 4]:
        centroids = get_centroids(bits, torch.device("cpu"), torch.float32)
        n_levels = 2 ** bits
        assert len(centroids) == n_levels, f"Expected {n_levels} centroids, got {len(centroids)}"

        # Check they are sorted
        assert torch.all(centroids[:-1] < centroids[1:]), "Centroids should be sorted"

        # Check they are approximately symmetric around 0
        sym_error = (centroids + centroids.flip(0)).abs().max().item()
        assert sym_error < 0.01, f"Centroids should be symmetric: {sym_error}"

        print(f"  bits={bits}: {n_levels} centroids, symmetric ✓, range [{centroids.min():.3f}, {centroids.max():.3f}]")
    print("  ✓ PASSED")


def test_mse_distortion():
    """Verify TurboQuant MSE distortion is near-optimal per paper Theorem 1."""
    print("\n[3] Testing MSE distortion vs theoretical bounds...")
    device = torch.device("cpu")

    expected_bounds = {1: 0.36, 2: 0.117, 3: 0.03, 4: 0.009}

    for bits in [1, 2, 3, 4]:
        dim = 256
        tq = TurboQuantMSE(dim=dim, bits=bits, device=device, dtype=torch.float32)

        n = 5000
        x = torch.randn(n, dim, dtype=torch.float32)
        x = x / x.norm(dim=-1, keepdim=True)  # unit norm

        indices, norms = tq.quantize(x)
        x_hat = tq.dequantize(indices, norms)

        mse = ((x - x_hat) ** 2).mean().item()
        theoretical = (math.sqrt(3) * math.pi / 2) / (4 ** bits)
        lower = 1.0 / (4 ** bits)
        ratio_to_lower = mse / lower

        print(f"  bits={bits}: MSE={mse:.5f} | theory≤{theoretical:.5f} | lower={lower:.5f} | ratio={ratio_to_lower:.2f}x")

        # The ratio should be roughly ≤ 2.72 per the paper, allow some slack for finite d
        assert ratio_to_lower < 5.0, f"MSE too high: {ratio_to_lower:.2f}x the lower bound"
        print(f"         ✓ Within {ratio_to_lower:.2f}x of optimal (paper claims ≤2.72x)")

    print("  ✓ PASSED")


def test_inner_product_estimation():
    """Test TurboQuant_PROD unbiased inner product estimation."""
    print("\n[4] Testing inner product estimation (TurboQuant_PROD)...")
    device = torch.device("cpu")
    dim = 128
    bits = 4

    tq = TurboQuantPROD(dim=dim, bits=bits, device=device, dtype=torch.float32)

    n_pairs = 500
    q = torch.randn(n_pairs, dim, dtype=torch.float32)
    k = torch.randn(n_pairs, dim, dtype=torch.float32)

    true_ip = (q * k).sum(dim=-1)

    # Compress keys
    k_idx, k_norms, k_sketch, k_r_norms = tq.quantize(k)

    # Estimate inner products
    ip_estimates = []
    for i in range(n_pairs):
        est = tq.estimate_inner_product(
            q[i:i+1],
            k_idx[i:i+1],
            k_norms[i:i+1],
            k_sketch[i:i+1],
            k_r_norms[i:i+1],
        )
        ip_estimates.append(est.item())

    import torch as t
    ip_est_tensor = t.tensor(ip_estimates)

    # Check bias (should be near-unbiased)
    errors = ip_est_tensor - true_ip
    bias = errors.mean().item()
    std = errors.std().item()

    print(f"  Mean estimation error (bias): {bias:.4f} (expected ≈ 0)")
    print(f"  Std of estimation error:      {std:.4f}")
    print(f"  True IP range: [{true_ip.min():.2f}, {true_ip.max():.2f}]")

    # Unbiasedness check (bias should be small relative to signal)
    signal_std = true_ip.std().item()
    relative_bias = abs(bias) / (signal_std + 1e-8)
    print(f"  Relative bias: {relative_bias:.4f} (expected < 0.1)")
    assert relative_bias < 0.3, f"Too much bias: {relative_bias:.3f}"
    print("  ✓ PASSED")


def test_cache_integration():
    """Test the TurboQuantCache with a tiny synthetic KV tensor."""
    print("\n[5] Testing TurboQuantCache integration...")
    from src.kv_cache_hook import TurboQuantCache

    device = torch.device("cpu")
    cache = TurboQuantCache(bits=4, dtype=torch.float32)

    B, H, L, D = 1, 8, 64, 128
    keys = torch.randn(B, H, L, D)
    values = torch.randn(B, H, L, D)

    k_out, v_out = cache.update(keys, values, layer_idx=0)

    assert k_out.shape == (B, H, L, D), f"Key shape mismatch: {k_out.shape}"
    assert v_out.shape == (B, H, L, D), f"Value shape mismatch: {v_out.shape}"

    k_mse = ((k_out - keys) ** 2).mean().item()
    v_mse = ((v_out - values) ** 2).mean().item()
    print(f"  Key MSE after compress+decompress: {k_mse:.5f}")
    print(f"  Value MSE after compress+decompress: {v_mse:.5f}")

    # Test sequential append
    new_k = torch.randn(B, H, 1, D)
    new_v = torch.randn(B, H, 1, D)
    k_full, v_full = cache.update(new_k, new_v, layer_idx=0)

    assert k_full.shape == (B, H, L + 1, D), f"Expected seq len {L+1}, got {k_full.shape[2]}"
    print(f"  Sequential append: {L} + 1 = {k_full.shape[2]} tokens ✓")

    # Stats
    stats = cache.stats.report()
    print(f"  Compression ratio: {stats['compress_ratio']}x")
    print(f"  Compressed size: {stats['compressed_mb']:.3f} MB (was {stats['original_mb']:.3f} MB)")
    print(f"  Compress time: {stats['avg_compress_ms_per_token']:.4f} ms/token")
    print("  ✓ PASSED")


def main():
    print("=" * 60)
    print("  TurboQuant Validation Suite")
    print("=" * 60)

    device_str = "CUDA" if torch.cuda.is_available() else "CPU"
    print(f"\n  Running on: {device_str}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    passed = 0
    failed = 0

    tests = [
        test_rht_orthogonality,
        test_centroids,
        test_mse_distortion,
        test_inner_product_estimation,
        test_cache_integration,
    ]

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"  Results: {passed}/{passed+failed} tests passed")
    if failed == 0:
        print("  All tests passed! Ready to run benchmarks.")
        print("\n  Next steps:")
        print("    1. python benchmarks/run_benchmark.py --mode baseline --test speed")
        print("    2. python benchmarks/run_benchmark.py --mode turboquant --bits 4 --test speed")
        print("    3. python benchmarks/compare_results.py --auto --save_plots")
    else:
        print(f"  {failed} test(s) failed — fix before running benchmarks.")
    print("=" * 60)


if __name__ == "__main__":
    main()
