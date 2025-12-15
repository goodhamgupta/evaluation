#!/usr/bin/env python3
"""
Generate model cards for quantized ColQwen3 models.
Reads quantization configs, performance data from JSON result files, and generates README.md files.
"""

import json
import re
from pathlib import Path
from typing import Optional

# Directory containing model directories
BASE_DIR = Path(__file__).parent

# Language mapping
LANGUAGE_MAP = {
    "eng-Latn": "English",
    "fra-Latn": "French",
    "spa-Latn": "Spanish",
    "deu-Latn": "German",
    "ita-Latn": "Italian",
    "por-Latn": "Portuguese",
}

# Model directories to process with their result folder names
MODEL_CONFIGS = {
    "tomoro-ai-colqwen3-embed-4b-autoawq-w4a16-seqlen-256": {
        "size": "4b",
        "seqlen": "256",
        "results_dir": "results_tomoro-ai-colqwen3-embed-4b-w4a16-autoawq-seqlen-256",
    },
    "tomoro-ai-colqwen3-embed-4b-w4a16-autoawq-seqlen-512": {
        "size": "4b",
        "seqlen": "512",
        "results_dir": "results_tomoro-ai-colqwen3-embed-4b-w4a16-autoawq-seqlen-512",
    },
    "tomoro-ai-colqwen3-embed-4b-w4a16-autoawq-seqlen-1024": {
        "size": "4b",
        "seqlen": "1024",
        "results_dir": "results_tomoro-ai-colqwen3-embed-4b-w4a16-autoawq-seqlen-1024",
    },
    "tomoro-ai-colqwen3-embed-8b-w4a16-autoawq-seqlen-256": {
        "size": "8b",
        "seqlen": "256",
        "results_dir": "results_tomoro-ai-colqwen3-embed-8b-w4a16-autoawq-seqlen-256",
    },
    "tomoro-ai-colqwen3-embed-8b-w4a16-autoawq-seqlen-512": {
        "size": "8b",
        "seqlen": "512",
        "results_dir": "results_w8a16_autoawq_vidore_full_eval",
    },
    "tomoro-ai-colqwen3-embed-8b-w4a16-autoawq-seqlen-1024": {
        "size": "8b",
        "seqlen": "1024",
        "results_dir": "results_tomoro-ai-colqwen3-embed-8b-w4a16-autoawq-seqlen-1024",
    },
}

# Memory usage from reports
MEMORY_INFO = {
    "4b": {
        "original_memory": "8.4 GB",
        "quantized_memory": "~3.5 GB",
        "original_params": "4.0B",
    },
    "8b": {
        "original_memory": "16.7 GB",
        "quantized_memory": "~7.9 GB",
        "original_params": "8.0B",
    },
}


def read_json_file(filepath: Path) -> Optional[dict]:
    """Read a JSON file and return its contents."""
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not read {filepath}: {e}")
        return None


def load_benchmark_results_from_folder(folder: Path, english_only: bool = False) -> dict[str, float]:
    """Load NDCG@5 scores for all benchmarks in a folder.

    Args:
        folder: Path to the results folder
        english_only: If True, only include English language results

    Returns:
        Dictionary mapping benchmark name (with language suffix) to NDCG@5 score
    """
    results = {}
    if not folder.exists():
        print(f"Warning: Results folder not found: {folder}")
        return results

    for json_file in folder.glob("*.json"):
        if json_file.name == "model_meta.json":
            continue
        try:
            with open(json_file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not read {json_file}: {e}")
            continue

        task_name = data.get("task_name", json_file.stem)

        for entry in data.get("scores", {}).get("test", []):
            ndcg_at_5 = entry.get("ndcg_at_5")
            if ndcg_at_5 is None:
                continue

            languages = entry.get("languages", [])
            if not languages:
                continue

            lang_code = languages[0]
            is_english = lang_code == "eng-Latn"
            lang_name = LANGUAGE_MAP.get(lang_code, lang_code)

            if english_only and not is_english:
                continue

            key = f"{task_name} [{lang_name}]"
            results[key] = ndcg_at_5

    return results


def parse_comparison_report(report_path: Path) -> dict:
    """Parse a comparison report markdown file to extract original scores and performance data.

    Returns dict with:
        - original_avg: float
        - seqlen_avgs: dict mapping seqlen to avg score
        - seqlen_deltas: dict mapping seqlen to delta string
        - seqlen_improved: dict mapping seqlen to count
        - seqlen_degraded: dict mapping seqlen to count
    """
    result = {
        "original_avg": 0.0,
        "seqlen_avgs": {},
        "seqlen_deltas": {},
        "seqlen_improved": {},
        "seqlen_degraded": {},
    }

    if not report_path.exists():
        return result

    with open(report_path, "r") as f:
        content = f.read()

    # Extract from Overall Scores table
    # Pattern: | Original (FP16) | 0.75798 | - |
    orig_match = re.search(r"\| Original \(FP16\) \| ([\d.]+) \|", content)
    if orig_match:
        result["original_avg"] = float(orig_match.group(1))

    # Pattern: | AWQ (seqlen=256) | 0.75688 | -0.14% |
    awq_pattern = r"\| AWQ \(seqlen=(\d+)\) \| ([\d.]+) \| ([+-]?[\d.]+%) \|"
    for match in re.finditer(awq_pattern, content):
        seqlen = match.group(1)
        avg = float(match.group(2))
        delta = match.group(3)
        result["seqlen_avgs"][seqlen] = avg
        result["seqlen_deltas"][seqlen] = delta

    # Extract improved/degraded counts from Performance by Calibration Sequence Length table
    # Pattern: | **Improved** | 11 | 1 | 9 |
    improved_match = re.search(r"\| \*\*Improved\*\* \| (\d+) \| (\d+) \| (\d+) \|", content)
    if improved_match:
        result["seqlen_improved"]["256"] = int(improved_match.group(1))
        result["seqlen_improved"]["512"] = int(improved_match.group(2))
        result["seqlen_improved"]["1024"] = int(improved_match.group(3))

    degraded_match = re.search(r"\| \*\*Degraded\*\* \| (\d+) \| (\d+) \| (\d+) \|", content)
    if degraded_match:
        result["seqlen_degraded"]["256"] = int(degraded_match.group(1))
        result["seqlen_degraded"]["512"] = int(degraded_match.group(2))
        result["seqlen_degraded"]["1024"] = int(degraded_match.group(3))

    return result


def get_all_performance_data() -> dict:
    """Load performance data from comparison reports for all model sizes."""
    data = {}

    for size in ["4B", "8B"]:
        size_lower = size.lower()
        data[size_lower] = {
            "english": parse_comparison_report(BASE_DIR / f"comparison_report_{size}_english.md"),
            "all_languages": parse_comparison_report(BASE_DIR / f"comparison_report_{size}_all_languages.md"),
        }

    return data


def generate_model_card(
    dir_name: str,
    model_config: dict,
    quant_config: dict,
    quant_metadata: dict,
    model_arch_config: dict,
    perf_data: dict,
    quantized_results: dict,
) -> str:
    """Generate the model card content."""

    size = model_config["size"]
    full_size = size.upper()
    seqlen = model_config["seqlen"]

    # Get memory info
    mem_info = MEMORY_INFO[size]

    # Get performance info from parsed comparison reports
    perf_all = perf_data[size]["all_languages"]
    perf_eng = perf_data[size]["english"]

    # Get original averages from comparison reports
    original_avg_all = perf_all.get("original_avg", 0)
    original_avg_eng = perf_eng.get("original_avg", 0)

    # Get quantized averages and deltas directly from comparison reports (for consistent benchmark sets)
    actual_avg_all = perf_all.get("seqlen_avgs", {}).get(seqlen, 0)
    actual_avg_eng = perf_eng.get("seqlen_avgs", {}).get(seqlen, 0)

    delta_all_str = perf_all.get("seqlen_deltas", {}).get(seqlen, "N/A")
    delta_eng_str = perf_eng.get("seqlen_deltas", {}).get(seqlen, "N/A")

    # Parse delta for quality retention calculation
    try:
        delta_all = float(delta_all_str.replace('%', '').replace('+', ''))
    except (ValueError, AttributeError):
        delta_all = 0

    # Get improved/degraded counts from comparison reports
    improved_all = perf_all.get("seqlen_improved", {}).get(seqlen, 0)
    degraded_all = perf_all.get("seqlen_degraded", {}).get(seqlen, 0)

    # Extract quantization details
    bits = quant_config.get("bits", 4)
    group_size = quant_config.get("group_size", 128)
    sym = quant_config.get("sym", True)
    iters = quant_config.get("iters", 1000)
    nsamples = quant_config.get("nsamples", 300)
    batch_size = quant_config.get("batch_size", 100)

    # Get metadata info
    calibration_dataset = quant_metadata.get("calibration_dataset", "NeelNanda/pile-10k")
    quantized_layers = quant_metadata.get("quantized_layers", 252)
    fp16_layers = quant_metadata.get("fp16_layers", 105 if size == "4b" else 117)

    # Model config details
    embed_dim = model_arch_config.get("embed_dim", 320)
    max_visual_tokens = model_arch_config.get("max_num_visual_tokens", 1280)

    # Determine the HuggingFace model ID format
    hf_model_id = f"shubhamg2208/{dir_name}"

    # Get all seqlen performance data for comparison table
    all_seqlen_data = {}
    for sl in ["256", "512", "1024"]:
        all_seqlen_data[sl] = {
            "avg": perf_all.get("seqlen_avgs", {}).get(sl, 0),
            "delta": perf_all.get("seqlen_deltas", {}).get(sl, "N/A"),
        }

    # Generate model card
    card = f"""---
license: apache-2.0
license_name: apache-2.0
license_link: https://www.apache.org/licenses/LICENSE-2.0
tags:
- text
- image
- video
- multimodal-embedding
- vidore
- colpali
- colqwen3
- multilingual-embedding
- quantized
- awq
- autoround
- w4a16
language:
- multilingual
library_name: transformers
pipeline_tag: visual-document-retrieval
base_model:
- TomoroAI/tomoro-colqwen3-embed-{size}
---

# {dir_name}

## Overview

This is a **W4A16 quantized** version of [TomoroAI/tomoro-colqwen3-embed-{size}](https://huggingface.co/TomoroAI/tomoro-colqwen3-embed-{size}), a state-of-the-art [ColPali](https://arxiv.org/abs/2407.01449)-style multimodal embedding model. The quantization was performed using [AutoRound](https://github.com/intel/auto-round) with AutoAWQ backend.

The quantized model achieves **{mem_info['quantized_memory']} memory usage** (vs {mem_info['original_memory']} for the original), enabling deployment on consumer GPUs while maintaining competitive retrieval performance.

## Model Details

| Property | Value |
|----------|-------|
| **Original Model** | [TomoroAI/tomoro-colqwen3-embed-{size}](https://huggingface.co/TomoroAI/tomoro-colqwen3-embed-{size}) |
| **Parameters** | {mem_info['original_params']} |
| **Quantization** | W4A16 (4-bit weights, 16-bit activations) |
| **Quantization Method** | AutoRound with AutoAWQ backend |
| **Calibration Sequence Length** | {seqlen} |
| **Memory Usage (Quantized)** | {mem_info['quantized_memory']} |
| **Memory Usage (Original)** | {mem_info['original_memory']} |
| **Embedding Dimension** | {embed_dim} |
| **Max Visual Tokens** | {max_visual_tokens} |

## Quantization Configuration

| Parameter | Value |
|-----------|-------|
| **Bits** | {bits} |
| **Group Size** | {group_size} |
| **Symmetric** | {sym} |
| **Calibration Dataset** | {calibration_dataset} |
| **Calibration Sequence Length** | {seqlen} |
| **Iterations** | {iters} |
| **Number of Samples** | {nsamples} |
| **Batch Size** | {batch_size} |
| **Quantized Layers** | {quantized_layers} |
| **FP16 Layers (Vision)** | {fp16_layers} |

> **Note:** Only the text tower (language model) is quantized. The vision encoder remains in FP16/BF16 to preserve visual feature quality.

## Performance

### NDCG@5 on ViDoRe Benchmark (All Languages)

| Model | Average NDCG@5 | Change |
|-------|----------------|--------|
| Original (FP16) | {original_avg_all:.5f} | - |
| **This Model (W4A16, seqlen={seqlen})** | **{actual_avg_all:.5f}** | **{delta_all_str}** |

### NDCG@5 on ViDoRe Benchmark (English Only)

| Model | Average NDCG@5 | Change |
|-------|----------------|--------|
| Original (FP16) | {original_avg_eng:.5f} | - |
| **This Model (W4A16, seqlen={seqlen})** | **{actual_avg_eng:.5f}** | **{delta_eng_str}** |

### Performance Summary

- **Benchmarks Improved:** {improved_all}
- **Benchmarks Degraded:** {degraded_all}
- **Overall Quality Retention:** ~{100 - abs(delta_all):.1f}%

## Memory Efficiency

The quantized model enables deployment on GPUs with limited memory:

| GPU Memory | Original Model | Quantized Model |
|------------|----------------|-----------------|
| 8 GB | {"Cannot fit" if size == "8b" else "Marginal"} | {"Fits with batch size ~64" if size == "4b" else "Cannot fit"} |
| 12 GB | {"Cannot fit" if size == "8b" else "Fits comfortably"} | {"Fits with batch size ~256" if size == "4b" else "Fits with batch size ~128"} |
| 16 GB | {"Marginal" if size == "8b" else "Fits comfortably"} | {"High batch sizes possible" if size == "4b" else "Fits with batch size ~512"} |
| 24 GB | Fits comfortably | High batch sizes possible |

## Usage

### Prerequisites

```bash
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install transformers pillow requests
pip install flash-attn --no-build-isolation  # Optional but recommended
```

### Inference Code

```python
import torch
from transformers import AutoModel, AutoProcessor
from PIL import Image
import requests
from io import BytesIO

# Configuration
MODEL_ID = "{hf_model_id}"
DTYPE = torch.bfloat16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load Model & Processor
processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    max_num_visual_tokens=1280,
)
model = AutoModel.from_pretrained(
    MODEL_ID,
    dtype=DTYPE,
    attn_implementation="sdpa",  # Use "flash_attention_2" if available
    trust_remote_code=True,
    device_map=DEVICE,
).eval()

# Sample queries and documents
queries = [
    "Retrieve the city of Singapore",
    "Retrieve the city of Beijing",
]
doc_urls = [
    "https://upload.wikimedia.org/wikipedia/commons/2/27/Singapore_skyline_2022.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/6/61/Beijing_skyline_at_night.JPG",
]

def load_image(url: str) -> Image.Image:
    headers = {{"User-Agent": "Mozilla/5.0"}}
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content)).convert("RGB")

def encode_queries(texts):
    batch = processor.process_texts(texts=texts)
    batch = {{k: v.to(DEVICE) for k, v in batch.items()}}
    with torch.inference_mode():
        out = model(**batch)
    return out.embeddings.to(torch.bfloat16).cpu()

def encode_docs(urls):
    images = [load_image(url) for url in urls]
    features = processor.process_images(images=images)
    features = {{k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in features.items()}}
    with torch.inference_mode():
        out = model(**features)
    return out.embeddings.to(torch.bfloat16).cpu()

# Encode and score
query_embeddings = encode_queries(queries)
doc_embeddings = encode_docs(doc_urls)
scores = processor.score_multi_vector(query_embeddings, doc_embeddings)
print(scores)
```

## Comparison with Other Calibration Lengths

| Calibration Length | Avg NDCG@5 | Delta | Best For |
|--------------------|------------|-------|----------|
| seqlen=256 | {all_seqlen_data['256']['avg']:.5f} | {all_seqlen_data['256']['delta']} | Short document retrieval |
| seqlen=512 | {all_seqlen_data['512']['avg']:.5f} | {all_seqlen_data['512']['delta']} | Balanced use cases |
| seqlen=1024 | {all_seqlen_data['1024']['avg']:.5f} | {all_seqlen_data['1024']['delta']} | Long document retrieval |

## Limitations

- **Reduced Precision:** 4-bit quantization introduces some accuracy loss compared to the original FP16 model.
- **Vision Encoder:** The vision encoder is not quantized to preserve visual feature quality.
- **Inference Backend:** Performance depends on the inference backend (AutoAWQ, vLLM, etc.).

## License

This model is released under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0), consistent with the original model.

## Acknowledgements

- **Original Model:** [TomoroAI/tomoro-colqwen3-embed-{size}](https://huggingface.co/TomoroAI/tomoro-colqwen3-embed-{size}) by [Tomoro AI](https://tomoro.ai/)
- **Quantization Tool:** [AutoRound](https://github.com/intel/auto-round) by Intel
- **Base Architecture:** [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-{full_size}-Instruct) by Alibaba

## Citation

If you use this model, please cite both the original model and this quantized version:

```bibtex
@misc{{huang2025beyond,
  author = {{Huang, Xin and Tan, Kye Min}},
  title = {{Beyond Text: Unlocking True Multimodal, End-to-end RAG with Tomoro ColQwen3}},
  year = {{2025}},
  url = {{https://tomoro.ai/insights/beyond-text-unlocking-true-multimodal-end-to-end-rag-with-tomoro-colqwen3}},
  publisher = {{Tomoro.ai}}
}}

@misc{{autoround,
  author = {{Intel Corporation}},
  title = {{AutoRound: Advanced Weight-Only Quantization Algorithm}},
  year = {{2024}},
  url = {{https://github.com/intel/auto-round}}
}}
```
"""

    return card


def main():
    """Main function to generate model cards."""

    print("Loading performance data from comparison reports...")
    perf_data = get_all_performance_data()

    for dir_name, model_config in MODEL_CONFIGS.items():
        model_dir = BASE_DIR / dir_name

        if not model_dir.exists():
            print(f"Warning: Directory {model_dir} does not exist, skipping...")
            continue

        print(f"Processing {dir_name}...")

        # Read config files
        quant_config = read_json_file(model_dir / "quantization_config.json") or {}
        quant_metadata = read_json_file(model_dir / "quantization_metadata.json") or {}
        model_arch_config = read_json_file(model_dir / "config.json") or {}

        # Load benchmark results from JSON files
        results_dir = model_dir / model_config["results_dir"]
        quantized_results = {
            "all_languages": load_benchmark_results_from_folder(results_dir, english_only=False),
            "english": load_benchmark_results_from_folder(results_dir, english_only=True),
        }

        print(f"  Loaded {len(quantized_results['all_languages'])} all-language benchmarks")
        print(f"  Loaded {len(quantized_results['english'])} English benchmarks")

        # Generate model card
        model_card = generate_model_card(
            dir_name,
            model_config,
            quant_config,
            quant_metadata,
            model_arch_config,
            perf_data,
            quantized_results,
        )

        # Write README.md
        readme_path = model_dir / "README.md"
        with open(readme_path, "w") as f:
            f.write(model_card)

        print(f"  Written: {readme_path}")

    print("\nDone! Model cards generated for all quantized models.")


if __name__ == "__main__":
    main()
