"""
TurboQuant Benchmark Runner
Runs baseline or TurboQuant-compressed inference and logs results.

Benchmark design rationale
---------------------------
TurboQuant compresses the *stored* KV cache — it does not reduce memory during
the prefill (prompt processing) phase.  During prefill each attention layer
must materialise the full Q @ Kᵀ attention matrix, which is O(seq²) and
independent of whether the cache is compressed.

On a 12 GB GPU (e.g. RTX 5070) the prefill of an 8 192-token prompt already
exceeds VRAM and spills into system RAM, making the baseline and the
compressed run equally slow and memory-heavy.  That comparison tells us
nothing about what the paper claims.

The correct setup is:
  - Prefill context ≤ 4 096 tokens (fits cleanly in 12 GB).
  - Generate 512+ new tokens so the accumulated KV cache grows large enough
    that compression matters.
  - Measure *end-of-generation* VRAM (model weights + KV cache, activations
    freed) to see the actual savings, in addition to peak VRAM.

Tests
-----
  needle      Needle-in-a-haystack: verifies quality is preserved.
  speed       Long-decode throughput and KV cache memory at various context
              lengths + generation lengths.
  perplexity  WikiText-2 perplexity: language-quality reference.

Usage:
    # Baseline (no quantization, full fp16 KV cache):
    python benchmarks/run_benchmark.py --mode baseline --test needle

    # TurboQuant with 4-bit compression:
    python benchmarks/run_benchmark.py --mode turboquant --bits 4 --test all

    # Full benchmark suite:
    python benchmarks/run_benchmark.py --mode turboquant --bits 4 --test all \\
        --context_lengths 512 1024 2048 4096 --n_decode_tokens 512

    # Quick smoke test:
    python benchmarks/run_benchmark.py --mode baseline --test speed \\
        --context_lengths 512 --n_decode_tokens 64
"""

import sys
import os
import json
import time
import argparse
import random
import datetime
import math
from typing import Tuple, List

import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer, AutoModelForCausalLM

# Llama 3.2 3B: 28 layers, 8 KV heads — same architecture family as the paper (Llama-3.1-8B).
# Requires accepting the license at huggingface.co/meta-llama/Llama-3.2-3B-Instruct
# and setting HF_TOKEN: set HF_TOKEN=hf_xxx  (Windows) or export HF_TOKEN=hf_xxx (Linux/Mac)
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(device: str = "cuda"):
    print(f"[*] Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    print(f"[*] Model loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    return model, tokenizer


# ---------------------------------------------------------------------------
# VRAM measurement
# ---------------------------------------------------------------------------

def get_vram_gb() -> float:
    """Current allocated VRAM — use after generation to read KV cache + model footprint."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e9
    return 0.0


def get_peak_vram_gb() -> float:
    """Peak VRAM since last reset — dominated by prefill activations."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e9
    return 0.0


def reset_peak_vram():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# ---------------------------------------------------------------------------
# Needle-in-a-haystack test
# ---------------------------------------------------------------------------

FILLER_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Scientists discovered a new species of butterfly in the Amazon rainforest.",
    "The stock market closed higher on Tuesday driven by tech sector gains.",
    "A local bakery won the national award for their sourdough bread recipe.",
    "Engineers at the university developed a more efficient solar panel design.",
    "The ancient ruins were discovered beneath the modern city streets.",
    "Weather forecasters predict temperatures will rise next week across the region.",
    "The new library opened its doors to the public with over 50,000 books.",
    "Researchers published findings on the role of sleep in memory consolidation.",
    "The city council voted to expand the public transportation network.",
    "A team of archaeologists unearthed artifacts from the Bronze Age.",
    "The documentary film won multiple awards at the international festival.",
    "Local farmers reported record harvests due to favorable weather conditions.",
    "The software update introduced several new features and performance improvements.",
    "Medical researchers announced progress in developing treatments for rare diseases.",
]

SECRET_CODE_TEMPLATE = "The secret passcode is: {code}."
QUESTION_TEMPLATE = "Based on the passage above, what is the secret passcode? Answer with only the code."


def build_needle_prompt(target_tokens: int, tokenizer, needle_position_frac: float = 0.5) -> Tuple[str, str]:
    """
    Build a prompt with a needle (secret code) inserted at needle_position_frac.
    Returns (prompt, expected_answer).
    """
    code = f"ALPHA{random.randint(1000, 9999)}BETA"
    needle = SECRET_CODE_TEMPLATE.format(code=code)

    # Build filler text until we reach approximately target_tokens
    filler_parts = []
    current_text = ""
    while True:
        sentence = random.choice(FILLER_SENTENCES)
        current_text += " " + sentence
        tokens = tokenizer.encode(current_text, add_special_tokens=False)
        if len(tokens) >= target_tokens - 50:  # leave room for needle + question
            break
        filler_parts.append(sentence)

    full_filler = " ".join(filler_parts)

    # Insert needle at the target position
    words = full_filler.split()
    insert_at = int(len(words) * needle_position_frac)
    words.insert(insert_at, needle)
    passage = " ".join(words)

    prompt = f"{passage}\n\n{QUESTION_TEMPLATE}"
    return prompt, code


def run_needle_test(
    model,
    tokenizer,
    context_lengths: list,
    n_trials: int = 20,
    use_turboquant: bool = False,
    bits: int = 4,
    device: str = "cuda",
) -> list:
    """
    Needle-in-a-haystack: measures accuracy at retrieving a hidden code from a
    long context.  Validates that TurboQuant compression does not hurt recall.

    Uses max_new_tokens=20 since the answer is a short code — this test is
    purely about quality, not decode speed.
    """
    from src.kv_cache_hook import run_with_cache

    results = []
    for ctx_len in context_lengths:
        print(f"\n  [Needle] Context length: {ctx_len} tokens, {n_trials} trials")
        correct = 0
        latencies = []
        vrams = []
        kv_cache_sizes = []

        for trial in range(n_trials):
            try:
                prompt, expected_code = build_needle_prompt(ctx_len, tokenizer)
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=ctx_len + 100).to(device)
                actual_len = inputs["input_ids"].shape[1]

                reset_peak_vram()
                output_ids, cache, timing = run_with_cache(
                    model,
                    inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    max_new_tokens=20,
                    use_turboquant=use_turboquant,
                    bits=bits,
                )

                # Decode only the new tokens
                new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
                answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

                is_correct = expected_code in answer
                correct += 1 if is_correct else 0
                latencies.append(timing["total_ms"])
                vrams.append(get_peak_vram_gb())

                if cache is not None:
                    kv_cache_sizes.append(cache.vram_usage_bytes() / 1e9)

                if trial % 5 == 0:
                    status = "✓" if is_correct else "✗"
                    print(f"    Trial {trial+1:2d}/{n_trials}: {status} | {actual_len} tok | "
                          f"{timing['total_ms']:.0f}ms | {get_peak_vram_gb():.2f}GB VRAM")

            except torch.cuda.OutOfMemoryError:
                print(f"    OOM at context {ctx_len} tokens — skipping remaining trials")
                break
            except Exception as e:
                print(f"    Error trial {trial}: {e}")
                continue

        if latencies:
            result = {
                "test": "needle",
                "context_length": ctx_len,
                "n_trials": len(latencies),
                "accuracy": round(correct / len(latencies), 4),
                "correct": correct,
                "avg_latency_ms": round(sum(latencies) / len(latencies), 1),
                "avg_peak_vram_gb": round(sum(vrams) / len(vrams), 3) if vrams else None,
                "avg_kv_cache_gb": round(sum(kv_cache_sizes) / len(kv_cache_sizes), 4) if kv_cache_sizes else None,
            }
            results.append(result)
            print(f"    → Accuracy: {result['accuracy']*100:.1f}% | "
                  f"Avg VRAM: {result['avg_peak_vram_gb']:.2f}GB | "
                  f"Avg KV: {result['avg_kv_cache_gb']}GB")

    return results


# ---------------------------------------------------------------------------
# Speed & KV cache memory test
# ---------------------------------------------------------------------------

def run_speed_test(
    model,
    tokenizer,
    context_lengths: list,
    n_decode_tokens: int = 512,
    n_repeats: int = 5,
    use_turboquant: bool = False,
    bits: int = 4,
    device: str = "cuda",
) -> list:
    """
    Measures decode throughput and KV cache memory footprint.

    Each run:
      1. Prefill: process a fixed-length prompt (context_length tokens).
      2. Decode: generate n_decode_tokens new tokens autoregressively.

    The KV cache grows to (context_length + n_decode_tokens) slots by the end.
    With n_decode_tokens=512 the cache is large enough to show meaningful
    savings between fp16 baseline and TurboQuant-compressed runs.

    Metrics reported:
      avg_peak_vram_gb  — peak during prefill (similar for both modes; dominated
                          by attention activations, not KV storage)
      avg_end_vram_gb   — VRAM after generation completes: model weights + KV
                          cache only (activations freed).  This is where
                          TurboQuant shows its memory savings.
      avg_kv_cache_gb   — exact size of the compressed cache tensor.
      avg_tokens_per_sec — overall throughput (prefill + decode).
      avg_decode_tps    — decode-only throughput (approximated as
                          n_decode_tokens / (total_ms - prefill_ms)).
    """
    from src.kv_cache_hook import run_with_cache

    # Fixed prompt prefix long enough to fill any requested context_length
    base_prompt = "The following is a detailed analysis of " + (" ".join(FILLER_SENTENCES * 100))

    results = []
    for ctx_len in context_lengths:
        print(f"\n  [Speed] Prefill: {ctx_len} tokens | Decode: {n_decode_tokens} new tokens")

        # Tokenize to exact context_length
        inputs = tokenizer(
            base_prompt, return_tensors="pt", truncation=True, max_length=ctx_len
        ).to(device)
        actual_len = inputs["input_ids"].shape[1]

        times = []
        tps_list = []
        decode_tps_list = []
        peak_vrams = []
        end_vrams = []
        kv_sizes = []

        # Measure prefill time once (same for baseline and TQ since both do
        # full attention over the prompt).
        prefill_ms_samples = []

        for rep in range(n_repeats):
            try:
                # --- prefill timing (forward pass only, no decode) ---
                reset_peak_vram()
                torch.cuda.synchronize()
                t_pre = time.perf_counter()
                with torch.no_grad():
                    _ = model(inputs["input_ids"])
                torch.cuda.synchronize()
                prefill_ms = (time.perf_counter() - t_pre) * 1000
                prefill_ms_samples.append(prefill_ms)

                # --- full generation (prefill + decode) ---
                reset_peak_vram()
                torch.cuda.synchronize()

                output_ids, cache, timing = run_with_cache(
                    model,
                    inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    max_new_tokens=n_decode_tokens,
                    use_turboquant=use_turboquant,
                    bits=bits,
                    do_sample=False,
                    temperature=1.0,
                )

                torch.cuda.synchronize()
                end_vram = get_vram_gb()   # model weights + KV cache; activations freed
                peak_vram = get_peak_vram_gb()

                total_ms = timing["total_ms"]
                decode_ms = max(total_ms - prefill_ms, 1.0)
                decode_tps = n_decode_tokens / (decode_ms / 1000)

                times.append(total_ms)
                tps_list.append(timing["tokens_per_sec"])
                decode_tps_list.append(decode_tps)
                peak_vrams.append(peak_vram)
                end_vrams.append(end_vram)
                if cache is not None:
                    kv_sizes.append(cache.vram_usage_bytes() / 1e9)

                print(f"    Rep {rep+1}/{n_repeats}: {actual_len}+{n_decode_tokens} tok | "
                      f"{total_ms:.0f}ms total | "
                      f"peak {peak_vram:.2f}GB | end {end_vram:.2f}GB VRAM")

            except torch.cuda.OutOfMemoryError:
                print(f"    OOM at context {ctx_len} — skipping")
                break

        if times:
            avg_prefill_ms = round(sum(prefill_ms_samples) / len(prefill_ms_samples), 1)
            result = {
                "test": "speed",
                "context_length": actual_len,
                "n_decode_tokens": n_decode_tokens,
                "n_repeats": len(times),
                "avg_total_ms": round(sum(times) / len(times), 1),
                "avg_prefill_ms": avg_prefill_ms,
                "avg_tokens_per_sec": round(sum(tps_list) / len(tps_list), 1),
                "avg_decode_tps": round(sum(decode_tps_list) / len(decode_tps_list), 1),
                "avg_peak_vram_gb": round(sum(peak_vrams) / len(peak_vrams), 3),
                "avg_end_vram_gb": round(sum(end_vrams) / len(end_vrams), 3),
                "avg_kv_cache_gb": round(sum(kv_sizes) / len(kv_sizes), 4) if kv_sizes else None,
            }
            results.append(result)
            print(f"    → decode {result['avg_decode_tps']} tok/s | "
                  f"peak VRAM {result['avg_peak_vram_gb']:.2f}GB | "
                  f"end VRAM {result['avg_end_vram_gb']:.2f}GB | "
                  f"KV {result['avg_kv_cache_gb']}GB")

    return results


# ---------------------------------------------------------------------------
# Perplexity test
# ---------------------------------------------------------------------------

def run_perplexity_test(
    model,
    tokenizer,
    context_lengths: list,
    use_turboquant: bool = False,
    bits: int = 4,
    device: str = "cuda",
    n_samples: int = 20,
) -> list:
    """Measure perplexity using WikiText-2 validation set."""
    print("\n  [Perplexity] Loading WikiText-2...")
    try:
        from datasets import load_dataset
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        text = "\n\n".join([x for x in dataset["text"] if len(x.strip()) > 100])
    except Exception as e:
        print(f"    Could not load WikiText-2: {e}. Using synthetic text.")
        text = " ".join(FILLER_SENTENCES * 500)

    all_tokens = tokenizer.encode(text)
    print(f"    Total tokens in eval set: {len(all_tokens)}")

    results = []
    for ctx_len in context_lengths:
        print(f"\n  [Perplexity] Context length: {ctx_len} tokens")

        nlls = []
        stride = ctx_len // 2
        start_positions = list(range(0, min(len(all_tokens) - ctx_len, n_samples * stride), stride))[:n_samples]

        for i, start in enumerate(start_positions):
            end = start + ctx_len
            chunk = all_tokens[start:end]
            input_ids = torch.tensor([chunk], dtype=torch.long, device=device)

            try:
                with torch.no_grad():
                    if use_turboquant:
                        from src.kv_cache_hook import apply_turboquant_to_model
                        cache = apply_turboquant_to_model(model, bits=bits, verbose=False)
                        outputs = model(input_ids, past_key_values=cache, use_cache=True, labels=input_ids)
                    else:
                        outputs = model(input_ids, use_cache=True, labels=input_ids)

                nll = outputs.loss.item()
                nlls.append(nll)

                if i % 5 == 0:
                    ppl_so_far = math.exp(sum(nlls) / len(nlls))
                    print(f"    Sample {i+1}/{len(start_positions)}: NLL={nll:.3f}, PPL so far={ppl_so_far:.2f}")

            except torch.cuda.OutOfMemoryError:
                print(f"    OOM at {ctx_len} tokens — skipping")
                break

        if nlls:
            avg_nll = sum(nlls) / len(nlls)
            ppl = math.exp(avg_nll)
            result = {
                "test": "perplexity",
                "context_length": ctx_len,
                "n_samples": len(nlls),
                "avg_nll": round(avg_nll, 4),
                "perplexity": round(ppl, 2),
            }
            results.append(result)
            print(f"    → Perplexity: {ppl:.2f}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="TurboQuant Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", choices=["baseline", "turboquant"], default="baseline",
                        help="baseline = full fp16 KV cache; turboquant = compressed KV cache")
    parser.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 8],
                        help="Quantization bits (only used in turboquant mode)")
    parser.add_argument("--test", default="all", choices=["all", "needle", "speed", "perplexity"],
                        help="Which test(s) to run")
    parser.add_argument("--context_lengths", type=int, nargs="+",
                        default=[512, 1024, 2048, 4096],
                        help="Prefill context lengths to benchmark. "
                             "8192 is intentionally excluded: at that length "
                             "prefill activations overflow 12 GB VRAM and the "
                             "comparison no longer reflects KV cache compression.")
    parser.add_argument("--n_trials", type=int, default=20,
                        help="Number of trials for needle test")
    parser.add_argument("--n_repeats", type=int, default=5,
                        help="Repeats for speed test")
    parser.add_argument("--n_decode_tokens", type=int, default=512,
                        help="New tokens to generate in the speed test. "
                             "512+ lets the KV cache grow large enough to show "
                             "meaningful memory savings from compression.")
    parser.add_argument("--device", default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--output_log", type=str, default=None,
                        help="Path to save JSON log (auto-generated if not set)")
    args = parser.parse_args()

    use_turboquant = args.mode == "turboquant"

    # Auto-generate log path
    if args.output_log is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        bits_tag = f"_{args.bits}bit" if use_turboquant else ""
        log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
        os.makedirs(log_dir, exist_ok=True)
        args.output_log = os.path.join(log_dir, f"{args.mode}{bits_tag}_{ts}.json")

    print(f"\n{'='*60}")
    print(f"  TurboQuant Benchmark")
    print(f"  Mode: {args.mode.upper()}" + (f" ({args.bits}-bit)" if use_turboquant else ""))
    print(f"  Tests: {args.test}")
    print(f"  Context lengths: {args.context_lengths}")
    print(f"  Decode tokens (speed test): {args.n_decode_tokens}")
    print(f"  Output: {args.output_log}")
    print(f"{'='*60}\n")

    model, tokenizer = load_model(args.device)

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    gpu_total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0

    log = {
        "run_id": f"{args.mode}_{args.bits}bit_{datetime.datetime.now().isoformat()}",
        "config": {
            "model": MODEL_ID,
            "mode": args.mode,
            "bits": args.bits if use_turboquant else 16,
            "device": args.device,
            "gpu": gpu_name,
            "gpu_total_gb": round(gpu_total_gb, 1),
            "context_lengths": args.context_lengths,
            "n_decode_tokens": args.n_decode_tokens,
        },
        "results": [],
        "summary": {},
    }

    # Run tests
    all_results = []

    if args.test in ("needle", "all"):
        print("\n[TEST 1/3] Needle-in-a-Haystack")
        needle_results = run_needle_test(
            model, tokenizer, args.context_lengths,
            n_trials=args.n_trials,
            use_turboquant=use_turboquant, bits=args.bits, device=args.device,
        )
        all_results.extend(needle_results)

    if args.test in ("speed", "all"):
        print("\n[TEST 2/3] Speed & KV Cache Memory")
        speed_results = run_speed_test(
            model, tokenizer, args.context_lengths,
            n_decode_tokens=args.n_decode_tokens,
            n_repeats=args.n_repeats,
            use_turboquant=use_turboquant, bits=args.bits, device=args.device,
        )
        all_results.extend(speed_results)

    if args.test in ("perplexity", "all"):
        print("\n[TEST 3/3] Perplexity")
        ppl_results = run_perplexity_test(
            model, tokenizer, args.context_lengths,
            use_turboquant=use_turboquant, bits=args.bits, device=args.device,
        )
        all_results.extend(ppl_results)

    log["results"] = all_results

    # Build summary
    needle_accs = [r["accuracy"] for r in all_results if r["test"] == "needle"]
    speed_tps = [r["avg_decode_tps"] for r in all_results if r["test"] == "speed"]
    ppls = [r["perplexity"] for r in all_results if r["test"] == "perplexity"]
    end_vrams = [r["avg_end_vram_gb"] for r in all_results if r["test"] == "speed" and r.get("avg_end_vram_gb")]
    kv_sizes = [r["avg_kv_cache_gb"] for r in all_results if r.get("avg_kv_cache_gb")]

    log["summary"] = {
        "avg_needle_accuracy": round(sum(needle_accs) / len(needle_accs), 4) if needle_accs else None,
        "avg_decode_tps": round(sum(speed_tps) / len(speed_tps), 1) if speed_tps else None,
        "avg_perplexity": round(sum(ppls) / len(ppls), 2) if ppls else None,
        "avg_end_vram_gb": round(sum(end_vrams) / len(end_vrams), 3) if end_vrams else None,
        "avg_kv_cache_gb": round(sum(kv_sizes) / len(kv_sizes), 4) if kv_sizes else None,
    }

    # Save log
    with open(args.output_log, "w") as f:
        json.dump(log, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  BENCHMARK COMPLETE")
    print(f"  Log saved: {args.output_log}")
    print(f"\n  Summary:")
    for k, v in log["summary"].items():
        if v is not None:
            print(f"    {k}: {v}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
