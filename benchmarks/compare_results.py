"""
TurboQuant Results Comparison
Loads multiple benchmark JSON logs and generates comparison tables, plots, and a report.

Usage:
    python benchmarks/compare_results.py logs/baseline_*.json logs/turboquant_4bit_*.json

    # Auto-discover all logs:
    python benchmarks/compare_results.py --auto

    # Save plots to results/:
    python benchmarks/compare_results.py --auto --save_plots
"""

import sys
import os
import json
import glob
import argparse
import datetime
from typing import List, Dict, Any, Optional

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_log(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def auto_discover_logs(log_dir: str = None) -> List[str]:
    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    return sorted(glob.glob(os.path.join(log_dir, "*.json")))


def get_results_by_test(log: dict, test_name: str) -> List[dict]:
    return [r for r in log.get("results", []) if r.get("test") == test_name]


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

def format_table(headers: list, rows: list, title: str = "") -> str:
    if not rows:
        return f"  (no data)\n"

    col_widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0)) for i, h in enumerate(headers)]
    sep = "  " + "-+-".join("-" * w for w in col_widths)
    header_row = "  " + " | ".join(str(h).ljust(col_widths[i]) for i, h in enumerate(headers))

    lines = []
    if title:
        lines.append(f"\n  {title}")
        lines.append("  " + "=" * len(title))
    lines.append(header_row)
    lines.append(sep)
    for row in rows:
        lines.append("  " + " | ".join(str(row[i]).ljust(col_widths[i]) for i, _ in enumerate(headers)))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

def delta_str(baseline_val, test_val, higher_is_better: bool = True, is_ratio: bool = False) -> str:
    if baseline_val is None or test_val is None:
        return "N/A"
    if baseline_val == 0:
        return "N/A"
    delta = test_val - baseline_val
    pct = (delta / abs(baseline_val)) * 100
    arrow = "▲" if delta > 0 else "▼"
    better = (delta > 0) == higher_is_better
    sign = "+" if delta > 0 else ""
    color_indicator = "[BETTER]" if better else "[WORSE]"
    return f"{sign}{pct:.1f}% {arrow} {color_indicator}"


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

def compare_logs(logs: List[dict], save_plots: bool = False, output_dir: str = None) -> str:
    """Generate a full comparison report."""

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(output_dir, exist_ok=True)

    # Identify baseline (mode=baseline or bits=16) and turboquant runs
    baseline_logs = [l for l in logs if l.get("config", {}).get("mode") == "baseline"]
    tq_logs = [l for l in logs if l.get("config", {}).get("mode") == "turboquant"]

    lines = []
    lines.append("=" * 70)
    lines.append("  TURBOQUANT BENCHMARK COMPARISON REPORT")
    lines.append(f"  Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)

    # Config table
    lines.append("\n## Run Configurations\n")
    cfg_headers = ["Run ID", "Mode", "Bits", "GPU", "Model"]
    cfg_rows = []
    for l in logs:
        cfg = l.get("config", {})
        run_id = l.get("run_id", "unknown")[:40]
        cfg_rows.append([run_id, cfg.get("mode", "?"), cfg.get("bits", "?"),
                         cfg.get("gpu", "?")[:20], cfg.get("model", "?")[-20:]])
    lines.append(format_table(cfg_headers, cfg_rows))

    # ------------------------------------------------------------------
    # Needle-in-a-Haystack comparison
    # ------------------------------------------------------------------
    lines.append("\n\n## Needle-in-a-Haystack Results\n")
    lines.append("  (Higher accuracy = better; lower VRAM = better)\n")

    needle_headers = ["Run", "Context", "Accuracy", "VRAM GB", "KV Cache GB", "Latency ms"]
    needle_rows = []

    for log in logs:
        run_label = f"{log['config']['mode']} {log['config']['bits']}b"
        for r in get_results_by_test(log, "needle"):
            needle_rows.append([
                run_label,
                r.get("context_length", "?"),
                f"{r.get('accuracy', 0)*100:.1f}%",
                f"{r.get('avg_peak_vram_gb', 'N/A')}",
                f"{r.get('avg_kv_cache_gb', 'N/A')}",
                f"{r.get('avg_latency_ms', 'N/A')}",
            ])
    lines.append(format_table(needle_headers, needle_rows))

    # Delta table (baseline vs each TQ run)
    if baseline_logs and tq_logs:
        lines.append("\n  Delta vs Baseline (Needle):\n")
        delta_headers = ["TQ Run", "Context", "Accuracy Δ", "VRAM Δ"]
        delta_rows = []
        baseline = baseline_logs[0]
        for tq_log in tq_logs:
            run_label = f"TQ {tq_log['config']['bits']}b"
            base_needle = {r["context_length"]: r for r in get_results_by_test(baseline, "needle")}
            for r in get_results_by_test(tq_log, "needle"):
                ctx = r["context_length"]
                b = base_needle.get(ctx, {})
                delta_rows.append([
                    run_label, ctx,
                    delta_str(b.get("accuracy"), r.get("accuracy"), higher_is_better=True),
                    delta_str(b.get("avg_peak_vram_gb"), r.get("avg_peak_vram_gb"), higher_is_better=False),
                ])
        lines.append(format_table(delta_headers, delta_rows))

    # ------------------------------------------------------------------
    # Speed & Memory comparison
    # ------------------------------------------------------------------
    lines.append("\n\n## Speed & Memory Results\n")
    lines.append("  (Higher tok/s = better; lower VRAM = better)\n")

    speed_headers = ["Run", "Context", "Tok/s", "VRAM GB", "KV Cache GB", "KV Compression"]
    speed_rows = []

    # Compute compression ratios
    for log in logs:
        run_label = f"{log['config']['mode']} {log['config']['bits']}b"
        bits = log["config"]["bits"]
        for r in get_results_by_test(log, "speed"):
            ctx = r.get("context_length", "?")
            kv_gb = r.get("avg_kv_cache_gb")
            # Theoretical compression ratio
            if log["config"]["mode"] == "turboquant" and kv_gb:
                ratio = f"{16/bits:.1f}x"
            elif log["config"]["mode"] == "baseline":
                ratio = "1.0x (fp16)"
            else:
                ratio = "N/A"
            speed_rows.append([
                run_label, ctx,
                f"{r.get('avg_tokens_per_sec', 'N/A')}",
                f"{r.get('avg_peak_vram_gb', 'N/A')}",
                f"{kv_gb or 'N/A'}",
                ratio,
            ])
    lines.append(format_table(speed_headers, speed_rows))

    # ------------------------------------------------------------------
    # Perplexity comparison
    # ------------------------------------------------------------------
    lines.append("\n\n## Perplexity Results\n")
    lines.append("  (Lower perplexity = better; baseline is reference)\n")

    ppl_headers = ["Run", "Context", "Perplexity", "NLL", "Δ vs Baseline"]
    ppl_rows = []

    base_ppls = {}
    if baseline_logs:
        base_ppls = {r["context_length"]: r for r in get_results_by_test(baseline_logs[0], "perplexity")}

    for log in logs:
        run_label = f"{log['config']['mode']} {log['config']['bits']}b"
        for r in get_results_by_test(log, "perplexity"):
            ctx = r["context_length"]
            base_ppl = base_ppls.get(ctx, {}).get("perplexity")
            if base_ppl and r.get("perplexity"):
                delta = f"+{r['perplexity'] - base_ppl:.3f}"
            else:
                delta = "N/A"
            ppl_rows.append([
                run_label, ctx,
                f"{r.get('perplexity', 'N/A')}",
                f"{r.get('avg_nll', 'N/A')}",
                delta,
            ])
    lines.append(format_table(ppl_headers, ppl_rows))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    lines.append("\n\n## Summary\n")
    for log in logs:
        s = log.get("summary", {})
        run_label = f"{log['config']['mode']} {log['config']['bits']}b"
        lines.append(f"  {run_label}:")
        if s.get("avg_needle_accuracy") is not None:
            lines.append(f"    Needle accuracy:  {s['avg_needle_accuracy']*100:.1f}%")
        if s.get("avg_tokens_per_sec") is not None:
            lines.append(f"    Tokens/sec:       {s['avg_tokens_per_sec']}")
        if s.get("avg_perplexity") is not None:
            lines.append(f"    Avg perplexity:   {s['avg_perplexity']}")
        if s.get("avg_peak_vram_gb") is not None:
            lines.append(f"    Peak VRAM:        {s['avg_peak_vram_gb']:.2f} GB")
        if s.get("avg_kv_cache_gb") is not None:
            lines.append(f"    KV cache:         {s['avg_kv_cache_gb']:.4f} GB")
        lines.append("")

    report = "\n".join(lines)
    print(report)

    # Save text report
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"comparison_{ts}.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n[*] Report saved: {report_path}")

    # Generate plots
    if save_plots:
        _generate_plots(logs, output_dir, ts)

    return report_path


def _generate_plots(logs: List[dict], output_dir: str, ts: str):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        print("[!] matplotlib not available, skipping plots")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("TurboQuant vs Baseline Benchmark Results", fontsize=14, fontweight="bold")

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for idx, log in enumerate(logs):
        mode = log["config"]["mode"]
        bits = log["config"]["bits"]
        label = f"{mode} {bits}b"
        color = colors[idx % len(colors)]

        # Plot 1: Needle accuracy vs context length
        ax = axes[0, 0]
        needle_data = sorted(get_results_by_test(log, "needle"), key=lambda r: r["context_length"])
        if needle_data:
            xs = [r["context_length"] for r in needle_data]
            ys = [r["accuracy"] * 100 for r in needle_data]
            ax.plot(xs, ys, marker="o", label=label, color=color)
        ax.set_title("Needle Accuracy vs Context Length")
        ax.set_xlabel("Context Length (tokens)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 105)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Peak VRAM vs context length
        ax = axes[0, 1]
        speed_data = sorted(get_results_by_test(log, "speed"), key=lambda r: r["context_length"])
        if speed_data:
            xs = [r["context_length"] for r in speed_data]
            ys = [r["avg_peak_vram_gb"] for r in speed_data if r.get("avg_peak_vram_gb")]
            if ys:
                ax.plot(xs[:len(ys)], ys, marker="s", label=label, color=color)
        ax.set_title("Peak VRAM vs Context Length")
        ax.set_xlabel("Context Length (tokens)")
        ax.set_ylabel("VRAM (GB)")
        ax.axhline(y=12, color="red", linestyle="--", alpha=0.5, label="12GB limit")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Tokens/sec vs context length
        ax = axes[1, 0]
        if speed_data:
            xs = [r["context_length"] for r in speed_data]
            ys = [r["avg_tokens_per_sec"] for r in speed_data if r.get("avg_tokens_per_sec")]
            if ys:
                ax.plot(xs[:len(ys)], ys, marker="^", label=label, color=color)
        ax.set_title("Tokens/sec vs Context Length")
        ax.set_xlabel("Context Length (tokens)")
        ax.set_ylabel("Tokens/sec")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Perplexity vs context length
        ax = axes[1, 1]
        ppl_data = sorted(get_results_by_test(log, "perplexity"), key=lambda r: r["context_length"])
        if ppl_data:
            xs = [r["context_length"] for r in ppl_data]
            ys = [r["perplexity"] for r in ppl_data]
            ax.plot(xs, ys, marker="D", label=label, color=color)
        ax.set_title("Perplexity vs Context Length")
        ax.set_xlabel("Context Length (tokens)")
        ax.set_ylabel("Perplexity")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"comparison_{ts}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"[*] Plot saved: {plot_path}")

    # Also save as fixed filename so README can always reference the latest
    latest_path = os.path.join(output_dir, "benchmark_comparison.png")
    plt.savefig(latest_path, dpi=150, bbox_inches="tight")
    print(f"[*] Latest plot updated: {latest_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare TurboQuant benchmark results")
    parser.add_argument("log_files", nargs="*", help="JSON log files to compare")
    parser.add_argument("--auto", action="store_true", help="Auto-discover all logs in logs/")
    parser.add_argument("--save_plots", action="store_true", help="Generate and save matplotlib plots")
    parser.add_argument("--output_dir", default=None, help="Directory for output files")
    args = parser.parse_args()

    log_paths = list(args.log_files)
    if args.auto:
        discovered = auto_discover_logs()
        log_paths.extend(discovered)
        if not discovered:
            print("[!] No logs found in logs/ directory. Run run_benchmark.py first.")
            return

    if not log_paths:
        print("Usage: compare_results.py <log1.json> <log2.json> ...")
        print("       compare_results.py --auto")
        return

    log_paths = list(dict.fromkeys(log_paths))  # deduplicate preserving order
    print(f"[*] Loading {len(log_paths)} log file(s)...")
    logs = [load_log(p) for p in log_paths]

    compare_logs(logs, save_plots=args.save_plots, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
