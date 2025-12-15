import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LANGUAGE_MAP = {
    "eng-Latn": "English",
    "fra-Latn": "French",
    "spa-Latn": "Spanish",
    "deu-Latn": "German",
    "ita-Latn": "Italian",
    "por-Latn": "Portuguese",
}

# Paths relative to mteb/results/results/ for original models
ORIGINAL_MODEL_PATHS = {
    "8B": "../results/results/TomoroAI__tomoro-colqwen3-embed-8b/5ba6feafdc0b61bfa2348989e80ac06ccf1a0a3f",
    "4B": "../results/results/TomoroAI__tomoro-colqwen3-embed-4b/4123a7add987edfa09105b9e420d91dffa10e9fd",
}

# Model configurations for quantized models (directories in reports/)
MODEL_VARIANTS = {
    "8B": {
        "original_memory_gb": 16.7,
        "quantized_memory_gb": 7.9,
        "seqlens": {
            256: {
                "model_dir": "tomoro-ai-colqwen3-embed-8b-w4a16-autoawq-seqlen-256",
                "results_dir": "results_tomoro-ai-colqwen3-embed-8b-w4a16-autoawq-seqlen-256",
            },
            512: {
                "model_dir": "tomoro-ai-colqwen3-embed-8b-w4a16-autoawq-seqlen-512",
                "results_dir": "results_w8a16_autoawq_vidore_full_eval",
            },
            1024: {
                "model_dir": "tomoro-ai-colqwen3-embed-8b-w4a16-autoawq-seqlen-1024",
                "results_dir": "results_tomoro-ai-colqwen3-embed-8b-w4a16-autoawq-seqlen-1024",
            },
        },
    },
    "4B": {
        "original_memory_gb": 8.4,
        "quantized_memory_gb": 3.5,
        "seqlens": {
            256: {
                "model_dir": "tomoro-ai-colqwen3-embed-4b-autoawq-w4a16-seqlen-256",
                "results_dir": "results_tomoro-ai-colqwen3-embed-4b-w4a16-autoawq-seqlen-256",
            },
            512: {
                "model_dir": "tomoro-ai-colqwen3-embed-4b-w4a16-autoawq-seqlen-512",
                "results_dir": "results_tomoro-ai-colqwen3-embed-4b-w4a16-autoawq-seqlen-512",
            },
            1024: {
                "model_dir": "tomoro-ai-colqwen3-embed-4b-w4a16-autoawq-seqlen-1024",
                "results_dir": "results_tomoro-ai-colqwen3-embed-4b-w4a16-autoawq-seqlen-1024",
            },
        },
    },
}


def load_model_meta(folder: Path) -> dict:
    """Load model metadata from a folder."""
    meta_path = folder / "model_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:  # noqa: PTH123
            return json.load(f)
    return {}


def load_quantization_config(folder: Path) -> dict:
    """Load quantization configuration from a model folder."""
    config_path = folder / "quantization_config.json"
    if config_path.exists():
        with open(config_path) as f:  # noqa: PTH123
            return json.load(f)
    return {}


def load_benchmark_results(
    folder: Path, english_only: bool = False
) -> dict[str, float]:
    """Load NDCG@5 scores for all benchmarks in a folder.

    Args:
        folder: Path to the results folder
        english_only: If True, only include English language results

    Returns:
        Dictionary mapping benchmark name (with language suffix) to NDCG@5 score
    """
    results = {}
    for json_file in folder.glob("*.json"):
        if json_file.name == "model_meta.json":
            continue
        with open(json_file) as f:  # noqa: PTH123
            data = json.load(f)
        task_name = data.get("task_name", json_file.stem)

        for entry in data["scores"]["test"]:
            ndcg_at_5 = entry.get("ndcg_at_5")
            if ndcg_at_5 is None:
                continue

            languages = entry.get("languages", [])

            lang_code = languages[0]
            is_english = lang_code == "eng-Latn"
            lang_name = LANGUAGE_MAP.get(lang_code, lang_code)

            if english_only and not is_english:
                continue

            key = f"{task_name} [{lang_name}]"
            results[key] = ndcg_at_5

    return results


def count_benchmark_files(folder: Path) -> int:
    """Count the number of benchmark JSON files in a folder."""
    return len([f for f in folder.glob("*.json") if f.name != "model_meta.json"])


def generate_performance_graph(
    original_results: dict[str, float],
    quantized_results_by_seqlen: dict[int, dict[str, float]],
    model_size: str,
    output_path: Path,
    english_only: bool = False,
) -> None:
    """Generate a bar chart comparing performance across seqlens."""
    # Find common benchmarks across all models
    all_keys = set(original_results.keys())
    for seqlen_results in quantized_results_by_seqlen.values():
        all_keys &= set(seqlen_results.keys())
    common_benchmarks = sorted(all_keys)

    if not common_benchmarks:
        print("No common benchmarks found for graph generation")
        return

    # Prepare data
    x = np.arange(len(common_benchmarks))
    width = 0.2

    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot original
    original_scores = [original_results[b] for b in common_benchmarks]
    ax.bar(x - 1.5 * width, original_scores, width, label="Original (FP16)", color="#1f77b4", edgecolor="white", linewidth=0.5)

    # Plot each seqlen - using colorblind-friendly palette
    colors = ["#2ca02c", "#ff7f0e", "#d62728"]  # green, orange, red
    seqlens = sorted(quantized_results_by_seqlen.keys())
    for i, seqlen in enumerate(seqlens):
        scores = [quantized_results_by_seqlen[seqlen][b] for b in common_benchmarks]
        ax.bar(x + (i - 0.5) * width, scores, width, label=f"AWQ (seqlen={seqlen})", color=colors[i], edgecolor="white", linewidth=0.5)

    # Shorten benchmark names for display
    short_names = [b.replace("Vidore", "").replace("Retrieval", "").replace(" [English]", "") for b in common_benchmarks]

    ax.set_xlabel("Benchmark", fontsize=12)
    ax.set_ylabel("NDCG@5 Score", fontsize=12)
    lang_suffix = "English Only" if english_only else "All Languages"
    ax.set_title(f"ColQwen3-Embed-{model_size}: Original vs AWQ Quantized ({lang_suffix})", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
    ax.legend(loc="upper left", fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Graph saved: {output_path}")


def generate_diff_graph(
    original_results: dict[str, float],
    quantized_results_by_seqlen: dict[int, dict[str, float]],
    model_size: str,
    output_path: Path,
    english_only: bool = False,
) -> None:
    """Generate a bar chart showing performance difference from original."""
    # Find common benchmarks across all models
    all_keys = set(original_results.keys())
    for seqlen_results in quantized_results_by_seqlen.values():
        all_keys &= set(seqlen_results.keys())
    common_benchmarks = sorted(all_keys)

    if not common_benchmarks:
        print("No common benchmarks found for diff graph generation")
        return

    x = np.arange(len(common_benchmarks))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 8))

    # Colorblind-friendly palette
    colors = ["#2ca02c", "#ff7f0e", "#d62728"]  # green, orange, red
    seqlens = sorted(quantized_results_by_seqlen.keys())

    for i, seqlen in enumerate(seqlens):
        diffs = []
        for b in common_benchmarks:
            orig = original_results[b]
            quant = quantized_results_by_seqlen[seqlen][b]
            pct_diff = ((quant - orig) / orig) * 100 if orig != 0 else 0
            diffs.append(pct_diff)
        ax.bar(x + (i - 1) * width, diffs, width, label=f"seqlen={seqlen}", color=colors[i], edgecolor="white", linewidth=0.5)

    # Shorten benchmark names for display
    short_names = [b.replace("Vidore", "").replace("Retrieval", "").replace(" [English]", "") for b in common_benchmarks]

    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.set_xlabel("Benchmark", fontsize=12)
    ax.set_ylabel("% Change from Original", fontsize=12)
    lang_suffix = "English Only" if english_only else "All Languages"
    ax.set_title(f"ColQwen3-Embed-{model_size}: Performance Difference ({lang_suffix})", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Diff graph saved: {output_path}")


def generate_avg_comparison_graph(
    all_results: dict[str, dict],
    output_path: Path,
    english_only: bool = False,
) -> None:
    """Generate a bar chart comparing average scores across all models and seqlens."""
    fig, ax = plt.subplots(figsize=(10, 6))

    model_sizes = ["4B", "8B"]
    seqlens = [256, 512, 1024]
    bar_width = 0.15
    x = np.arange(len(model_sizes))

    # Original scores
    original_avgs = []
    for ms in model_sizes:
        if ms in all_results and "original_avg" in all_results[ms]:
            original_avgs.append(all_results[ms]["original_avg"])
        else:
            original_avgs.append(0)

    ax.bar(x - 2 * bar_width, original_avgs, bar_width, label="Original (FP16)", color="#1f77b4", edgecolor="white", linewidth=0.5)

    # Quantized scores per seqlen - colorblind-friendly palette
    colors = ["#2ca02c", "#ff7f0e", "#d62728"]  # green, orange, red
    for i, seqlen in enumerate(seqlens):
        avgs = []
        for ms in model_sizes:
            key = f"seqlen_{seqlen}_avg"
            if ms in all_results and key in all_results[ms]:
                avgs.append(all_results[ms][key])
            else:
                avgs.append(0)
        ax.bar(x + (i - 1) * bar_width, avgs, bar_width, label=f"AWQ (seqlen={seqlen})", color=colors[i], edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Model Size", fontsize=12)
    ax.set_ylabel("Average NDCG@5 Score", fontsize=12)
    lang_suffix = "English Only" if english_only else "All Languages"
    ax.set_title(f"ColQwen3-Embed: Average Performance Comparison ({lang_suffix})", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(model_sizes)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Average comparison graph saved: {output_path}")


def generate_report(
    original_folder: Path,
    quantized_folders: dict[int, Path],
    quantization_configs: dict[int, dict],
    english_only: bool = False,
    model_size: str = "8B",
    original_memory_gb: float = 0.0,
    quantized_memory_gb: float = 0.0,
) -> tuple[str, dict]:
    """Generate markdown report comparing original and quantized models.

    Args:
        original_folder: Path to original model results
        quantized_folders: Dict mapping seqlen to Path of quantized model results
        quantization_configs: Dict mapping seqlen to quantization config
        english_only: If True, only include English language results
        model_size: Model size variant (e.g., "8B", "4B")
        original_memory_gb: Memory usage of original model in GB
        quantized_memory_gb: Memory usage of quantized model in GB

    Returns:
        Tuple of (markdown report string, summary dict for graphs)
    """
    original_meta = load_model_meta(original_folder)
    original_results = load_benchmark_results(original_folder, english_only=english_only)
    original_file_count = count_benchmark_files(original_folder)

    # Load results for each seqlen
    quantized_results_by_seqlen = {}
    for seqlen, folder in sorted(quantized_folders.items()):
        quantized_results_by_seqlen[seqlen] = load_benchmark_results(folder, english_only=english_only)

    # Find common benchmarks across all models
    all_keys = set(original_results.keys())
    for seqlen_results in quantized_results_by_seqlen.values():
        all_keys &= set(seqlen_results.keys())
    common_benchmarks = sorted(all_keys)

    lines = []
    lang_suffix = "English Only" if english_only else "All Languages"
    lines.append(f"# Model Comparison Report: {model_size} Original vs AWQ Quantized ({lang_suffix})")
    lines.append("")

    # Model Information
    lines.append("## Model Information")
    lines.append("")
    lines.append("| Property | Original (FP16) |")
    lines.append("|----------|-----------------|")
    if original_meta:
        lines.append(f"| **Model Name** | {original_meta.get('name', 'N/A')} |")
        n_params = original_meta.get('n_parameters', 0)
        lines.append(f"| **Parameters** | {n_params / 1e9:.1f}B |")
        lines.append(f"| **Memory Usage** | {original_memory_gb:.1f} GB |")
        lines.append(f"| **Release Date** | {original_meta.get('release_date', 'N/A')} |")
    lines.append("")

    # Quantization Configuration Section
    lines.append("## Quantization Configuration")
    lines.append("")
    lines.append("All quantized models use **AutoRound with AutoAWQ** backend, calibrated with **NeelNanda/pile-10k** dataset.")
    lines.append("")
    lines.append("| Property | seqlen=256 | seqlen=512 | seqlen=1024 |")
    lines.append("|----------|------------|------------|-------------|")

    # Extract common properties
    seqlens = sorted(quantized_folders.keys())
    config_props = ["bits", "group_size", "sym", "iters", "nsamples", "batch_size", "quant_method", "provider"]

    for prop in config_props:
        row = f"| **{prop}** |"
        for seqlen in seqlens:
            cfg = quantization_configs.get(seqlen, {})
            val = cfg.get(prop, "N/A")
            row += f" {val} |"
        lines.append(row)

    lines.append("")
    lines.append(f"**Quantized Memory Usage:** ~{quantized_memory_gb:.1f} GB")
    lines.append("")

    # Performance comparison table
    lines.append("## NDCG@5 Performance Comparison")
    lines.append("")

    header = "| Benchmark | Original |"
    sep = "|-----------|----------|"
    for seqlen in seqlens:
        header += f" seqlen={seqlen} | Δ% |"
        sep += "------------|-----|"
    lines.append(header)
    lines.append(sep)

    total_original = 0
    totals_by_seqlen = {s: 0 for s in seqlens}

    for benchmark in common_benchmarks:
        orig = original_results[benchmark]
        total_original += orig
        row = f"| {benchmark} | {orig:.5f} |"
        for seqlen in seqlens:
            quant = quantized_results_by_seqlen[seqlen][benchmark]
            totals_by_seqlen[seqlen] += quant
            diff = quant - orig
            pct_change = (diff / orig) * 100 if orig != 0 else 0
            if pct_change > 0:
                change_str = f"+{pct_change:.2f}%"
            elif pct_change < 0:
                change_str = f"{pct_change:.2f}%"
            else:
                change_str = "0.00%"
            row += f" {quant:.5f} | {change_str} |"
        lines.append(row)

    # Averages
    n_benchmarks = len(common_benchmarks)
    summary = {}
    if n_benchmarks > 0:
        avg_orig = total_original / n_benchmarks
        summary["original_avg"] = avg_orig

        avg_row = "| **Average** | **{:.5f}** |".format(avg_orig)
        for seqlen in seqlens:
            avg_quant = totals_by_seqlen[seqlen] / n_benchmarks
            summary[f"seqlen_{seqlen}_avg"] = avg_quant
            avg_diff = avg_quant - avg_orig
            avg_pct = (avg_diff / avg_orig) * 100 if avg_orig != 0 else 0
            if avg_pct > 0:
                change_str = f"+{avg_pct:.2f}%"
            elif avg_pct < 0:
                change_str = f"{avg_pct:.2f}%"
            else:
                change_str = "0.00%"
            avg_row += f" **{avg_quant:.5f}** | **{change_str}** |"
        lines.append("|" + "-" * 11 + "|" + "----------|" + ("------------|-----|" * len(seqlens)))
        lines.append(avg_row)

    lines.append("")

    # Summary statistics
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Benchmark files (Original):** {original_file_count}")
    lines.append(f"- **Total entries evaluated:** {n_benchmarks}")
    lines.append("")

    lines.append("### Performance by Calibration Sequence Length")
    lines.append("")
    lines.append("| Metric | seqlen=256 | seqlen=512 | seqlen=1024 |")
    lines.append("|--------|------------|------------|-------------|")

    for seqlen in seqlens:
        improved = sum(1 for b in common_benchmarks if quantized_results_by_seqlen[seqlen][b] > original_results[b])
        degraded = sum(1 for b in common_benchmarks if quantized_results_by_seqlen[seqlen][b] < original_results[b])
        summary[f"seqlen_{seqlen}_improved"] = improved
        summary[f"seqlen_{seqlen}_degraded"] = degraded

    improved_row = "| **Improved** |"
    degraded_row = "| **Degraded** |"
    unchanged_row = "| **Unchanged** |"

    for seqlen in seqlens:
        improved = summary.get(f"seqlen_{seqlen}_improved", 0)
        degraded = summary.get(f"seqlen_{seqlen}_degraded", 0)
        unchanged = n_benchmarks - improved - degraded
        improved_row += f" {improved} |"
        degraded_row += f" {degraded} |"
        unchanged_row += f" {unchanged} |"

    lines.append(improved_row)
    lines.append(degraded_row)
    lines.append(unchanged_row)

    lines.append("")
    lines.append("### Overall Scores")
    lines.append("")
    lines.append("| Model | Average NDCG@5 | Change from Original |")
    lines.append("|-------|----------------|----------------------|")
    lines.append(f"| Original (FP16) | {avg_orig:.5f} | - |")

    for seqlen in seqlens:
        avg_quant = summary.get(f"seqlen_{seqlen}_avg", 0)
        avg_pct = ((avg_quant - avg_orig) / avg_orig) * 100 if avg_orig != 0 else 0
        lines.append(f"| AWQ (seqlen={seqlen}) | {avg_quant:.5f} | {avg_pct:+.2f}% |")

    lines.append("")

    # Add graph references
    lines.append("## Performance Graphs")
    lines.append("")
    suffix = "english" if english_only else "all_languages"
    lines.append(f"![Performance Comparison](performance_comparison_{model_size}_{suffix}.png)")
    lines.append("")
    lines.append(f"![Performance Difference](performance_diff_{model_size}_{suffix}.png)")
    lines.append("")

    return "\n".join(lines), summary


def generate_reports_for_variant(
    reports_dir: Path, model_size: str, verbose: bool = True
) -> dict:
    """Generate reports for a specific model variant.

    Args:
        reports_dir: Base directory containing result folders
        model_size: Model size variant (e.g., "8B", "4B")
        verbose: If True, print reports to console

    Returns:
        Summary dict with average scores for each seqlen, or empty dict if failed
    """
    if model_size not in MODEL_VARIANTS:
        print(f"Unknown model variant: {model_size}")
        return {}

    config = MODEL_VARIANTS[model_size]
    original_folder = (reports_dir / ORIGINAL_MODEL_PATHS[model_size]).resolve()

    # Check if original folder exists
    if not original_folder.exists():
        print(f"Original folder not found: {original_folder}")
        return {}

    # Collect quantized folders and configs
    quantized_folders = {}
    quantization_configs = {}

    for seqlen, seqlen_config in config["seqlens"].items():
        model_dir = reports_dir / seqlen_config["model_dir"]
        results_dir = model_dir / seqlen_config["results_dir"]

        if not results_dir.exists():
            print(f"Quantized results folder not found: {results_dir}")
            continue

        quantized_folders[seqlen] = results_dir
        quantization_configs[seqlen] = load_quantization_config(model_dir)

    if not quantized_folders:
        print(f"No quantized result folders found for {model_size}")
        return {}

    print(f"\n{'=' * 80}")
    print(f"Generating reports for {model_size} model variant")
    print(f"{'=' * 80}")

    original_memory_gb = config["original_memory_gb"]
    quantized_memory_gb = config["quantized_memory_gb"]

    all_summaries = {}

    # Generate English-only report
    english_report, english_summary = generate_report(
        original_folder,
        quantized_folders,
        quantization_configs,
        english_only=True,
        model_size=model_size,
        original_memory_gb=original_memory_gb,
        quantized_memory_gb=quantized_memory_gb,
    )
    english_output_path = reports_dir / f"comparison_report_{model_size}_english.md"
    with open(english_output_path, "w") as f:  # noqa: PTH123
        f.write(english_report)
    print(f"English-only report generated: {english_output_path}")
    all_summaries["english"] = english_summary

    # Generate all languages report
    all_langs_report, all_langs_summary = generate_report(
        original_folder,
        quantized_folders,
        quantization_configs,
        english_only=False,
        model_size=model_size,
        original_memory_gb=original_memory_gb,
        quantized_memory_gb=quantized_memory_gb,
    )
    all_langs_output_path = reports_dir / f"comparison_report_{model_size}_all_languages.md"
    with open(all_langs_output_path, "w") as f:  # noqa: PTH123
        f.write(all_langs_report)
    print(f"All languages report generated: {all_langs_output_path}")
    all_summaries["all_languages"] = all_langs_summary

    # Load results for graphs
    original_results_english = load_benchmark_results(original_folder, english_only=True)
    original_results_all = load_benchmark_results(original_folder, english_only=False)

    quantized_results_english = {}
    quantized_results_all = {}
    for seqlen, folder in quantized_folders.items():
        quantized_results_english[seqlen] = load_benchmark_results(folder, english_only=True)
        quantized_results_all[seqlen] = load_benchmark_results(folder, english_only=False)

    # Generate graphs
    generate_performance_graph(
        original_results_english,
        quantized_results_english,
        model_size,
        reports_dir / f"performance_comparison_{model_size}_english.png",
        english_only=True,
    )
    generate_performance_graph(
        original_results_all,
        quantized_results_all,
        model_size,
        reports_dir / f"performance_comparison_{model_size}_all_languages.png",
        english_only=False,
    )
    generate_diff_graph(
        original_results_english,
        quantized_results_english,
        model_size,
        reports_dir / f"performance_diff_{model_size}_english.png",
        english_only=True,
    )
    generate_diff_graph(
        original_results_all,
        quantized_results_all,
        model_size,
        reports_dir / f"performance_diff_{model_size}_all_languages.png",
        english_only=False,
    )

    if verbose:
        print(f"\n{'=' * 80}")
        print(f"{model_size} ENGLISH ONLY REPORT:")
        print(f"{'=' * 80}\n")
        print(english_report)

        print(f"\n{'=' * 80}")
        print(f"{model_size} ALL LANGUAGES REPORT:")
        print(f"{'=' * 80}\n")
        print(all_langs_report)

    return all_summaries


def main():
    parser = argparse.ArgumentParser(
        description="Generate comparison reports for original vs quantized models"
    )
    parser.add_argument(
        "--variant",
        "-v",
        choices=list(MODEL_VARIANTS.keys()),
        default=None,
        help="Specific model variant to generate reports for (e.g., 8B, 4B). "
        "If not specified, generates reports for all available variants.",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress printing reports to console",
    )
    args = parser.parse_args()

    reports_dir = Path(__file__).parent

    if args.variant:
        variants = [args.variant]
    else:
        variants = list(MODEL_VARIANTS.keys())

    all_results = {}
    generated_count = 0

    for variant in variants:
        summaries = generate_reports_for_variant(reports_dir, variant, verbose=not args.quiet)
        if summaries:
            all_results[variant] = summaries.get("all_languages", {})
            generated_count += 1

    # Generate combined comparison graph
    if generated_count > 1 and all_results:
        generate_avg_comparison_graph(
            all_results,
            reports_dir / "performance_avg_comparison_all.png",
            english_only=False,
        )

    if generated_count == 0:
        print("\nNo reports generated. Make sure result folders exist.")
    else:
        print(f"\nGenerated reports for {generated_count} model variant(s).")


if __name__ == "__main__":
    main()
