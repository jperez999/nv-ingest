# VLM Image Captioning for nemo_retriever

Generate natural-language captions for document page images and element crops
using the **NVIDIA Nemotron-Nano-VL-12B-v2** Vision-Language Model.

## Overview

The VLM captioning stage sits between extraction/OCR and embedding in the
ingestion pipeline. It processes page images (and optionally cropped
tables/charts/infographics) and produces a `vlm_caption` column containing
generated captions.

Two inference modes are supported:

| Mode | Description | GPU Required |
|------|-------------|-------------|
| **Remote** | Connects to a vLLM or NIM HTTP endpoint | No (on client) |
| **Local** | Loads the HuggingFace model in-process | Yes (H100 80GB) |

## Quick Start

### Remote mode (recommended for production)

**1. Serve the model with vLLM:**

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model nvidia/Nemotron-Nano-VL-12B-V2-FP8 \
  --trust-remote-code \
  --quantization modelopt
```

**2. Use in the pipeline:**

```python
from nemo_retriever import create_ingestor
from nemo_retriever.params import VLMCaptionParams, ExtractParams

# Batch engine
ingestor = create_ingestor(run_mode="batch")
results = (
    ingestor
    .files("documents/*.pdf")
    .extract(ExtractParams(extract_text=True))
    .caption(VLMCaptionParams(
        invoke_url="http://localhost:8000/v1/chat/completions",
        prompt="Describe this document page in detail.",
    ))
    .embed()
    .ingest()
)

# Inprocess engine
ingestor = create_ingestor(run_mode="inprocess")
results = (
    ingestor
    .files("documents/*.pdf")
    .extract(ExtractParams(extract_text=True))
    .caption(VLMCaptionParams(
        invoke_url="http://localhost:8000/v1/chat/completions",
        prompt="Describe this document page in detail.",
    ))
    .embed()
    .ingest()
)
```

### Local HuggingFace mode

**1. Install dependencies:**

```bash
pip install causal_conv1d "transformers>=5.0" torch timm "mamba-ssm==2.2.5" \
  accelerate open_clip_torch numpy pillow
```

**2. Use in the pipeline (no `invoke_url` needed):**

```python
from nemo_retriever import create_ingestor
from nemo_retriever.params import VLMCaptionParams, ExtractParams

# Batch engine — model loaded inside Ray actor, 1 GPU per actor
ingestor = create_ingestor(run_mode="batch")
results = (
    ingestor
    .files("documents/*.pdf")
    .extract(ExtractParams(extract_text=True))
    .caption(VLMCaptionParams(
        prompt="Describe this document page in detail.",
    ))
    .embed()
    .ingest()
)

# Inprocess engine — model loaded in main process
ingestor = create_ingestor(run_mode="inprocess")
results = (
    ingestor
    .files("documents/*.pdf")
    .extract(ExtractParams(extract_text=True))
    .caption(VLMCaptionParams(
        prompt="Describe this document page in detail.",
    ))
    .embed()
    .ingest()
)
```

## Configuration Parameters

`VLMCaptionParams` accepts the following parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `invoke_url` | `str \| None` | `None` | Remote VLM endpoint URL. When set, uses remote inference. |
| `api_key` | `str \| None` | `None` | Auth token for remote endpoint. |
| `model_name` | `str` | `"nvidia/nemotron-nano-12b-v2-vl"` | Model name for remote inference. |
| `prompt` | `str` | `"Caption the content of this image:"` | Caption generation prompt. |
| `system_prompt` | `str \| None` | `"/no_think"` | System prompt (e.g. `"/no_think"` to disable thinking). |
| `max_tokens` | `int` | `512` | Maximum response tokens. |
| `temperature` | `float` | `1.0` | Sampling temperature. |
| `caption_page_images` | `bool` | `True` | Caption full page images. |
| `caption_tables` | `bool` | `False` | Caption detected table crops. |
| `caption_charts` | `bool` | `False` | Caption detected chart crops. |
| `caption_infographics` | `bool` | `False` | Caption detected infographic crops. |
| `request_timeout_s` | `float` | `120.0` | HTTP timeout for remote mode. |
| `device` | `str \| None` | `None` | GPU device for local mode (e.g. `"cuda:0"`). |
| `hf_cache_dir` | `str \| None` | `None` | HuggingFace cache directory for local mode. |
| `remote_retry` | `RemoteRetryParams` | defaults | Retry configuration for remote mode. |

## Batch Engine Details

In the batch engine, `VLMCaptioningActor` is used as a Ray Data `map_batches`
stage with `ActorPoolStrategy`. The actor loads the model once per worker and
processes batches of pages.

Additional batch-specific kwargs (passed via `**kwargs` to `.caption()`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `caption_batch_size` | `4` | Ray Data batch size. |
| `caption_workers` | `1` | ActorPool size. |

**GPU allocation:**
- Remote mode (`invoke_url` set): `num_gpus=0`
- Local mode (no `invoke_url`): `num_gpus=1`

## Inprocess Engine Details

In the inprocess engine, `vlm_caption_images()` is called directly as a
pipeline task function. When using local mode, the model is loaded once
in the main process and shared across all pages.

## Output Format

The captioning stage adds a `vlm_caption` column to the DataFrame. Each row
contains a dict with the following structure:

```python
{
    "page_caption": "A detailed caption of the page...",
    "element_captions": [
        {
            "type": "table",
            "bbox_xyxy_norm": [0.1, 0.2, 0.8, 0.9],
            "caption": "A table showing quarterly revenue..."
        },
        # ... more element captions
    ],
    "timing": {"seconds": 2.5},
    "error": None  # or error dict on failure
}
```

## Pipeline Placement

The captioning stage should be placed **after extraction/OCR** (so page images
and element detections are available) and **before embedding** (so captions can
be embedded alongside text):

```
.files() → .extract() → .caption() → .embed() → .vdb_upload() → .ingest()
```

## Element Captioning

When `caption_tables`, `caption_charts`, or `caption_infographics` are enabled,
the captioner will crop detected regions from the page image using bounding
boxes from the `page_elements_v3` column (produced by the extraction stage).
Each crop is captioned individually and stored in `element_captions`.

This requires that extraction was run with the corresponding `extract_*` flags:

```python
ingestor.extract(ExtractParams(
    extract_text=True,
    extract_tables=True,
    extract_charts=True,
))
.caption(VLMCaptionParams(
    caption_tables=True,
    caption_charts=True,
))
```

## Direct Usage

The core function and actor can be used independently:

```python
from nemo_retriever.vlm_captioning import vlm_caption_images, VLMCaptioningActor

# Direct function call (inprocess-style)
result_df = vlm_caption_images(
    pages_df,
    invoke_url="http://localhost:8000/v1/chat/completions",
    prompt="Describe this image.",
)

# Actor instantiation (for Ray Data)
actor = VLMCaptioningActor(
    invoke_url="http://localhost:8000/v1/chat/completions",
    prompt="Describe this image.",
)
result_df = actor(pages_df)
```

## Local Model Wrapper

The `NemotronNanoVL12BV2` class can also be used standalone:

```python
from nemo_retriever.model.local import NemotronNanoVL12BV2

model = NemotronNanoVL12BV2(device="cuda:0")
caption = model.caption_single(image_b64, prompt="Describe this image.")
captions = model.caption([img1_b64, img2_b64], prompt="Describe this image.")
```
