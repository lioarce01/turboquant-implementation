#!/usr/bin/env bash
# Full TurboQuant benchmark pipeline
# Run from the project root: bash run_full_benchmark.sh

set -e
echo "============================================================"
echo "  TurboQuant Full Benchmark Pipeline"
echo "============================================================"

# Step 0: Validate the algorithm
echo ""
echo "[Step 0] Validating TurboQuant implementation..."
python validate_turboquant.py

# Step 1: Baseline (full fp16 KV cache)
echo ""
echo "[Step 1] Running BASELINE (fp16 KV cache)..."
python benchmarks/run_benchmark.py \
    --mode baseline \
    --test all \
    --context_lengths 512 1024 2048 4096 8192 \
    --n_trials 20 \
    --n_repeats 5

# Step 2: TurboQuant 4-bit
echo ""
echo "[Step 2] Running TURBOQUANT 4-bit..."
python benchmarks/run_benchmark.py \
    --mode turboquant \
    --bits 4 \
    --test all \
    --context_lengths 512 1024 2048 4096 8192 \
    --n_trials 20 \
    --n_repeats 5

# Step 3: TurboQuant 3-bit (more aggressive compression)
echo ""
echo "[Step 3] Running TURBOQUANT 3-bit..."
python benchmarks/run_benchmark.py \
    --mode turboquant \
    --bits 3 \
    --test needle,speed \
    --context_lengths 512 1024 2048 4096 8192 \
    --n_trials 20 \
    --n_repeats 5

# Step 4: Compare and generate report
echo ""
echo "[Step 4] Generating comparison report..."
python benchmarks/compare_results.py --auto --save_plots

echo ""
echo "============================================================"
echo "  DONE. Check results/ directory for report and plots."
echo "============================================================"
