#!/usr/bin/env python3
"""
Generate model cards for quantized ColQwen3 models.
Reads quantization configs, performance data, and generates README.md files.
"""

import json
import os
import re
from pathlib import Path
from typing import Optional

# Directory containing model directories
BASE_DIR = Path(__file__).parent

# Model directories to process
MODEL_DIRS = [
    "tomoro-ai-colqwen3-embed-4b-autoawq-w4a16-seqlen-256",
    "tomoro-ai-colqwen3-embed-4b-w4a16-autoawq-seqlen-512",
    "tomoro-ai-colqwen3-embed-4b-w4a16-autoawq-seqlen-1024",
    "tomoro-ai-colqwen3-embed-8b-w4a16-autoawq-seqlen-256",
    "tomoro-ai-colqwen3-embed-8b-w4a16-autoawq-seqlen-512",
    "tomoro-ai-colqwen3-embed-8b-w4a16-autoawq-seqlen-1024",
]

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

# Performance data extracted from comparison reports (All Languages)
PERFORMANCE_DATA = {
    "4b": {
        "256": {"avg_ndcg5": 0.69611, "delta": "-0.59%", "improved": 10, "degraded": 30},
        "512": {"avg_ndcg5": 0.69696, "delta": "-0.47%", "improved": 16, "degraded": 24},
        "1024": {"avg_ndcg5": 0.69768, "delta": "-0.36%", "improved": 17, "degraded": 23},
        "original": 0.70023,
    },
    "8b": {
        "256": {"avg_ndcg5": 0.64044, "delta": "-0.32%", "improved": 28, "degraded": 43},
        "512": {"avg_ndcg5": 0.63063, "delta": "-1.84%", "improved": 14, "degraded": 56},
        "1024": {"avg_ndcg5": 0.64198, "delta": "-0.08%", "improved": 30, "degraded": 41},
        "original": 0.64247,
    },
}

# English-only performance data
PERFORMANCE_DATA_ENGLISH = {
    "4b": {
        "256": {"avg_ndcg5": 0.74545, "delta": "-0.26%", "improved": 5, "degraded": 16},
        "512": {"avg_ndcg5": 0.74432, "delta": "-0.42%", "improved": 7, "degraded": 14},
        "1024": {"avg_ndcg5": 0.74582, "delta": "-0.21%", "improved": 11, "degraded": 10},
        "original": 0.74743,
    },
    "8b": {
        "256": {"avg_ndcg5": 0.75688, "delta": "-0.14%", "improved": 11, "degraded": 11},
        "512": {"avg_ndcg5": 0.74782, "delta": "-1.34%", "improved": 1, "degraded": 20},
        "1024": {"avg_ndcg5": 0.75660, "delta": "-0.18%", "improved": 9, "degraded": 13},
        "original": 0.75798,
    },
}


def parse_model_name(dir_name: str) -> dict:
    """Parse model directory name to extract model info."""
    # Extract size (4b or 8b)
    if "4b" in dir_name.lower():
        size = "4b"
        full_size = "4B"
    else:
        size = "8b"
        full_size = "8B"

    # Extract seqlen
    seqlen_match = re.search(r"seqlen-(\d+)", dir_name)
    seqlen = seqlen_match.group(1) if seqlen_match else "256"

    return {
        "size": size,
        "full_size": full_size,
        "seqlen": seqlen,
        "dir_name": dir_name,
    }


def read_json_file(filepath: Path) -> Optional[dict]:
    """Read a JSON file and return its contents."""
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not read {filepath}: {e}")
        return None


def generate_model_card(model_info: dict, quant_config: dict, quant_metadata: dict, model_config: dict) -> str:
    """Generate the model card content."""

    size = model_info["size"]
    full_size = model_info["full_size"]
    seqlen = model_info["seqlen"]
    dir_name = model_info["dir_name"]

    # Get memory info
    mem_info = MEMORY_INFO[size]

    # Get performance info
    perf = PERFORMANCE_DATA[size][seqlen]
    perf_eng = PERFORMANCE_DATA_ENGLISH[size].get(seqlen, perf)
    original_score = PERFORMANCE_DATA[size]["original"]
    original_score_eng = PERFORMANCE_DATA_ENGLISH[size]["original"]

    # Extract quantization details
    bits = quant_config.get("bits", 4)
    group_size = quant_config.get("group_size", 128)
    sym = quant_config.get("sym", True)
    quant_method = quant_config.get("quant_method", "awq")
    iters = quant_config.get("iters", 1000)
    nsamples = quant_config.get("nsamples", 300)
    batch_size = quant_config.get("batch_size", 100)

    # Get metadata info
    calibration_dataset = quant_metadata.get("calibration_dataset", "NeelNanda/pile-10k")
    original_model = quant_metadata.get("original_model", f"TomoroAI/tomoro-colqwen3-embed-{size}")
    quantized_layers = quant_metadata.get("quantized_layers", 252)
    fp16_layers = quant_metadata.get("fp16_layers", 105)

    # Model config details
    embed_dim = model_config.get("embed_dim", 320)
    max_visual_tokens = model_config.get("max_num_visual_tokens", 1280)
    text_hidden_size = model_config.get("text_config", {}).get("hidden_size", 2560)
    num_layers = model_config.get("text_config", {}).get("num_hidden_layers", 36)

    # Determine the HuggingFace model ID format
    hf_model_id = f"shubhamg2208/{dir_name}"

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
| Original (FP16) | {original_score:.5f} | - |
| **This Model (W4A16, seqlen={seqlen})** | **{perf['avg_ndcg5']:.5f}** | **{perf['delta']}** |

### NDCG@5 on ViDoRe Benchmark (English Only)

| Model | Average NDCG@5 | Change |
|-------|----------------|--------|
| Original (FP16) | {original_score_eng:.5f} | - |
| **This Model (W4A16, seqlen={seqlen})** | **{perf_eng['avg_ndcg5']:.5f}** | **{perf_eng['delta']}** |

### Performance Summary

- **Benchmarks Improved:** {perf['improved']}
- **Benchmarks Degraded:** {perf['degraded']}
- **Overall Quality Retention:** ~{100 - abs(float(perf['delta'].replace('%', ''))):.1f}%

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
| seqlen=256 | {PERFORMANCE_DATA[size]['256']['avg_ndcg5']:.5f} | {PERFORMANCE_DATA[size]['256']['delta']} | Short document retrieval |
| seqlen=512 | {PERFORMANCE_DATA[size]['512']['avg_ndcg5']:.5f} | {PERFORMANCE_DATA[size]['512']['delta']} | Balanced use cases |
| seqlen=1024 | {PERFORMANCE_DATA[size]['1024']['avg_ndcg5']:.5f} | {PERFORMANCE_DATA[size]['1024']['delta']} | Long document retrieval |

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

    for dir_name in MODEL_DIRS:
        model_dir = BASE_DIR / dir_name

        if not model_dir.exists():
            print(f"Warning: Directory {model_dir} does not exist, skipping...")
            continue

        print(f"Processing {dir_name}...")

        # Parse model info from directory name
        model_info = parse_model_name(dir_name)

        # Read config files
        quant_config = read_json_file(model_dir / "quantization_config.json") or {}
        quant_metadata = read_json_file(model_dir / "quantization_metadata.json") or {}
        model_config = read_json_file(model_dir / "config.json") or {}

        # Generate model card
        model_card = generate_model_card(model_info, quant_config, quant_metadata, model_config)

        # Write README.md
        readme_path = model_dir / "README.md"
        with open(readme_path, "w") as f:
            f.write(model_card)

        print(f"  Written: {readme_path}")

    print("\nDone! Model cards generated for all quantized models.")


if __name__ == "__main__":
    main()
