#!/usr/bin/env bash
# Full TurboQuant benchmark pipeline — Llama-3.2-3B-Instruct on RTX 5070 12GB
# Run from the project root: bash run_full_benchmark.sh
#
# Requires a HuggingFace token with access to meta-llama/Llama-3.2-3B-Instruct:
#   export HF_TOKEN=hf_your_token_here
#
# Benchmark design notes
# ----------------------
# TurboQuant compresses the *stored* KV cache, not the prefill activations.
# On a 12 GB GPU, a 8 192-token prefill overflows VRAM and spills to system RAM,
# making the baseline and compressed runs equally slow — that comparison shows
# nothing about KV cache compression.
#
# The correct setup:
#   context_lengths ≤ 4096  — prefill fits cleanly in 12 GB.
#   n_decode_tokens 512     — long generation so the KV cache grows large
#                             enough to show meaningful savings.
#
# Key metrics to compare:
#   end_vram_gb   — VRAM after generation (model + KV cache, activations freed).
#                   This is where TurboQuant saves memory.
#   avg_decode_tps — decode-phase throughput.
#   accuracy / perplexity — quality preserved under compression.

set -e

if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN is not set."
    echo "  export HF_TOKEN=hf_your_token_here"
    exit 1
fi

echo "============================================================"
echo "  TurboQuant Full Benchmark Pipeline"
echo "  Model: meta-llama/Llama-3.2-3B-Instruct"
echo "  Context lengths: 512 1024 2048 4096"
echo "  Decode tokens (speed test): 512"
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
    --context_lengths 512 1024 2048 4096 \
    --n_trials 20 \
    --n_repeats 5 \
    --n_decode_tokens 512

# Step 2: TurboQuant 4-bit
echo ""
echo "[Step 2] Running TURBOQUANT 4-bit..."
python benchmarks/run_benchmark.py \
    --mode turboquant \
    --bits 4 \
    --test all \
    --context_lengths 512 1024 2048 4096 \
    --n_trials 20 \
    --n_repeats 5 \
    --n_decode_tokens 512

# Step 3: TurboQuant 3-bit (more aggressive compression)
echo ""
echo "[Step 3] Running TURBOQUANT 3-bit..."
python benchmarks/run_benchmark.py \
    --mode turboquant \
    --bits 3 \
    --test all \
    --context_lengths 512 1024 2048 4096 \
    --n_trials 20 \
    --n_repeats 5 \
    --n_decode_tokens 512

# Step 4: Compare and generate report
echo ""
echo "[Step 4] Generating comparison report and plots..."
python benchmarks/compare_results.py --auto --save_plots

echo ""
echo "============================================================"
echo "  DONE. Check results/ for report and benchmark_comparison.png"
echo "============================================================"
