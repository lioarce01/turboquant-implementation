#!/usr/bin/env bash
# Full TurboQuant benchmark pipeline — Llama-3.2-3B-Instruct on RTX 5070 12GB
# Run from the project root: bash run_full_benchmark.sh
#
# Requires a HuggingFace token with access to meta-llama/Llama-3.2-3B-Instruct:
#   export HF_TOKEN=hf_your_token_here

set -e

if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN is not set."
    echo "  export HF_TOKEN=hf_your_token_here"
    exit 1
fi

echo "============================================================"
echo "  TurboQuant Full Benchmark Pipeline"
echo "  Model: meta-llama/Llama-3.2-3B-Instruct"
echo "============================================================"

# Step 0: Validate the algorithm (no model needed)
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
    --test all \
    --context_lengths 512 1024 2048 4096 8192 \
    --n_trials 20 \
    --n_repeats 5

# Step 4: Compare and generate report
echo ""
echo "[Step 4] Generating comparison report and plots..."
python benchmarks/compare_results.py --auto --save_plots

echo ""
echo "============================================================"
echo "  DONE. Check results/ for report and benchmark_comparison.png"
echo "============================================================"
