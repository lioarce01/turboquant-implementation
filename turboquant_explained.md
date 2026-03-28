# TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate

**Paper:** https://arxiv.org/abs/2504.19874
**Authors:** Amir Zandieh, Majid Daliri, Majid Hadian, Vahab Mirrokni
**Published:** April 28, 2025

---

## What Is This Paper About?

When LLMs process long sequences, they store every token's **Key** and **Value** tensors in a "KV cache" — this cache grows linearly with context length and can easily consume several GB of GPU memory. TurboQuant is a compression algorithm that shrinks this cache 4–5× while keeping model output essentially identical.

The core idea: instead of storing full-precision (fp16) vectors, compress them to 3–4 bits per number using a mathematically near-optimal quantization scheme.

---

## The Problem It Solves

| Problem | Consequence |
|---------|-------------|
| KV cache grows with context length | Limits how long a context you can run |
| Existing quantization loses too much info | Degrades model quality |
| Existing methods need calibration data | Can't be applied online/streaming |
| Existing methods are slow to index | Adds latency overhead |

TurboQuant solves all four: it is **online** (no training data needed), **fast** (near-zero indexing overhead), and **near-optimal** (within a ~2.7× constant of the theoretical information-theoretic limit).

---

## Core Idea in Plain English

Imagine you have a high-dimensional vector (e.g., a 128-dim Key vector from an attention head). You want to store it in fewer bits.

**Step 1 — Random Rotation**
Multiply the vector by a random rotation matrix. This "scrambles" the vector so that all coordinates become approximately equal in magnitude and nearly independent of each other. After rotation, each coordinate looks like it was drawn from a known distribution (approximately Gaussian).

**Step 2 — Scalar Quantization**
Because coordinates are now approximately i.i.d. with a known distribution, you can apply the theoretically optimal scalar quantizer (Lloyd-Max quantizer) independently to each coordinate. This means finding the best set of `2^bits` centroids for that distribution and rounding each coordinate to the nearest centroid.

**Step 3 — Rotate Back**
To reconstruct, look up the centroid values and apply the inverse rotation. The error introduced is near-optimal by theory.

```
Original vector x (fp16, d dims)
       ↓  random rotation Π
Rotated vector x̃ (spread out, near-Gaussian)
       ↓  scalar quantize each coordinate to b bits
Indices (b bits per coordinate)
       ↓  dequantize: lookup centroids
Reconstructed x̃ (approx)
       ↓  inverse rotation Π⁻¹
Reconstructed x (approx, much smaller storage)
```

---

## Two Variants

### TurboQuant-MSE (Algorithm 1)
Minimizes **mean squared error** between original and reconstructed vector.
Best for: storing values in KV cache where you want the reconstructed vector to be close.

### TurboQuant-PROD (Algorithm 2)
Minimizes **inner product error** — preserves dot products between vectors.
Best for: attention scores (which are computed as dot products between Q and K vectors).

It works in two stages:
1. Apply TurboQuant-MSE with (b-1) bits → get approximate vector
2. Compute residual (what's left after step 1)
3. Apply **1-bit Quantized Johnson-Lindenstrauss (QJL)** transform on the residual → recovers unbiased inner product estimate with 1 extra bit

---

## Why Near-Optimal?

The paper proves a **lower bound**: any quantizer that stores b bits per coordinate must have MSE distortion ≥ 1/4^b.

TurboQuant achieves MSE ≤ (√3π/2) × (1/4^b) ≈ 2.72 × (1/4^b).

So TurboQuant is within a factor of **2.72× of the theoretical best possible quantizer**, using a data-oblivious (no calibration) approach.

---

## Experimental Results

All tests on Llama-3.1-8B-Instruct:

### KV Cache Compression (Needle-in-a-Haystack)
| Method | Recall Score | Compression |
|--------|-------------|-------------|
| Full cache (fp16) | 1.000 | 1× |
| SnapKV | 0.858 | 4× |
| KIVI | 0.981 | 4× |
| PolarQuant | 0.995 | 4× |
| **TurboQuant** | **0.997** | **4×** |

### LongBench (diverse long-context tasks)
| Method | Avg Score | Bits/element |
|--------|-----------|-------------|
| Full cache | 50.06 | 16 |
| TurboQuant | **50.06** | **3.5** |
| TurboQuant (mixed) | 49.44 | 2.5 |

**Zero quality loss at 3.5 bits** — a 4.6× compression over fp16.

### Indexing Speed
TurboQuant: 0.001–0.002 seconds
Product Quantization: 37–494 seconds
RabitQ: 597–3957 seconds

TurboQuant is **~100,000× faster to index** than competitors.

---

## Why It Works for LLMs Specifically

LLM attention keys and values are high-dimensional (e.g., 128-dim per head). After the random rotation:
- Coordinates become approximately i.i.d. Gaussian (by a concentration-of-measure argument)
- The distribution is known ahead of time (depends only on vector norm, not content)
- So the optimal quantizer is fixed and pre-computable — no calibration needed

This is the key insight that makes TurboQuant **data-oblivious** and **online**: the random rotation "normalizes" the input structure so a universal quantizer works.

---

## What TurboQuant Is NOT

- It's not a neural network or learned compression
- It's not model quantization (it doesn't touch model weights)
- It's not pruning or sparse attention
- It is specifically a **KV cache compression** method applied during inference

---

## Random Rotation Implementation

For efficiency, the paper uses a **Randomized Hadamard Transform (RHT)**:
1. Multiply each dimension by a random ±1 sign
2. Apply the Fast Walsh-Hadamard Transform (O(d log d) instead of O(d²))
3. Normalize by 1/√d

This achieves the same statistical properties as a full random rotation but is much faster.

---

## Summary

TurboQuant = **Random Rotation** + **Pre-computed Optimal Scalar Quantizer** + **(optional) 1-bit QJL for inner products**

Result: **4–5× smaller KV cache, essentially free in terms of model quality, and fast enough for real-time inference.**
