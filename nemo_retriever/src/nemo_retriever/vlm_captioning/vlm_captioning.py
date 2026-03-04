# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
VLM image captioning for nemo_retriever.

Provides ``vlm_caption_images()`` (the core DataFrame-level function) and
``VLMCaptioningActor`` (a Ray Data-compatible actor) that generate
natural-language captions for page images and/or cropped element regions using
the NVIDIA Nemotron-Nano-VL model.

Supports two inference modes:
- **Local**: Loads ``NemotronNanoVL12BV2`` from HuggingFace and runs on-device.
- **Remote**: Calls a vLLM / NIM HTTP endpoint via ``_generate_captions``
  from ``nv_ingest_api``.
"""

from __future__ import annotations

import time
import traceback
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from nemo_retriever.params import RemoteRetryParams

# Reuse helpers from ocr module for image cropping and encoding.
from nemo_retriever.ocr.ocr import _crop_all_from_page, _np_rgb_to_b64_png


def vlm_caption_images(
    batch_df: Any,
    *,
    model: Any = None,
    invoke_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model_name: str = "nvidia/nemotron-nano-12b-v2-vl",
    prompt: str = "Caption the content of this image:",
    system_prompt: Optional[str] = "/no_think",
    max_tokens: int = 512,
    temperature: float = 1.0,
    caption_page_images: bool = True,
    caption_tables: bool = False,
    caption_charts: bool = False,
    caption_infographics: bool = False,
    request_timeout_s: float = 120.0,
    remote_retry: Optional[RemoteRetryParams] = None,
    **kwargs: Any,
) -> Any:
    """
    Caption page images (and optionally element crops) using a VLM.

    For each row in *batch_df* that contains a ``page_image`` column with
    ``image_b64``, this function generates captions and stores them in a new
    ``vlm_caption`` column.

    Parameters
    ----------
    batch_df : pd.DataFrame
        Input DataFrame with ``page_image`` (dict with ``image_b64`` key).
    model : NemotronNanoVL12BV2 | None
        Local HuggingFace model instance. Mutually exclusive with *invoke_url*.
    invoke_url : str | None
        Remote VLM endpoint URL (vLLM OpenAI-compatible or NIM).
    api_key : str | None
        Auth token for remote endpoint.
    model_name : str
        Model name for remote inference.
    prompt : str
        Caption prompt.
    system_prompt : str | None
        Optional system prompt (e.g. ``"/no_think"``).
    max_tokens : int
        Maximum response tokens.
    temperature : float
        Sampling temperature.
    caption_page_images : bool
        Whether to caption full page images.
    caption_tables, caption_charts, caption_infographics : bool
        Whether to caption detected element crops.
    request_timeout_s : float
        HTTP timeout for remote mode.
    remote_retry : RemoteRetryParams | None
        Retry configuration for remote mode.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with an added ``vlm_caption`` column containing
        per-row dicts with ``page_caption``, ``element_captions``, ``timing``,
        and ``error`` keys.
    """
    if not isinstance(batch_df, pd.DataFrame):
        raise NotImplementedError("vlm_caption_images currently only supports pandas.DataFrame input.")

    invoke_url = (invoke_url or "").strip()
    use_remote = bool(invoke_url)
    if not use_remote and model is None:
        raise ValueError("A local `model` is required when `invoke_url` is not provided.")

    # Determine which element types to crop and caption.
    wanted_labels: set[str] = set()
    if caption_tables:
        wanted_labels.add("table")
    if caption_charts:
        wanted_labels.add("chart")
    if caption_infographics:
        wanted_labels.add("infographic")

    all_captions: List[Dict[str, Any]] = []
    t0_total = time.perf_counter()

    for row in batch_df.itertuples(index=False):
        page_caption: Optional[str] = None
        element_captions: List[Dict[str, Any]] = []
        row_error: Any = None

        try:
            page_image = getattr(row, "page_image", None) or {}
            page_image_b64 = page_image.get("image_b64") if isinstance(page_image, dict) else None
            if not isinstance(page_image_b64, str) or not page_image_b64:
                all_captions.append(
                    {"page_caption": None, "element_captions": [], "timing": None, "error": None}
                )
                continue

            # 1. Caption the full page image.
            if caption_page_images:
                page_caption = _caption_single(
                    image_b64=page_image_b64,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    model=model,
                    use_remote=use_remote,
                    invoke_url=invoke_url,
                    api_key=api_key,
                    model_name=model_name,
                    temperature=temperature,
                )

            # 2. Caption element crops (tables, charts, infographics).
            if wanted_labels:
                pe = getattr(row, "page_elements_v3", None)
                dets: List[Dict[str, Any]] = []
                if isinstance(pe, dict):
                    dets = pe.get("detections") or []
                if not isinstance(dets, list):
                    dets = []

                crops = _crop_all_from_page(page_image_b64, dets, wanted_labels)
                for label_name, bbox, crop_array in crops:
                    crop_b64 = _np_rgb_to_b64_png(crop_array)
                    cap = _caption_single(
                        image_b64=crop_b64,
                        prompt=prompt,
                        system_prompt=system_prompt,
                        model=model,
                        use_remote=use_remote,
                        invoke_url=invoke_url,
                        api_key=api_key,
                        model_name=model_name,
                        temperature=temperature,
                    )
                    element_captions.append(
                        {"type": label_name, "bbox_xyxy_norm": bbox, "caption": cap}
                    )

        except BaseException as e:
            print(f"Warning: VLM captioning failed: {type(e).__name__}: {e}")
            row_error = {
                "stage": "vlm_caption_images",
                "type": e.__class__.__name__,
                "message": str(e),
                "traceback": "".join(traceback.format_exception(type(e), e, e.__traceback__)),
            }

        all_captions.append(
            {
                "page_caption": page_caption,
                "element_captions": element_captions,
                "timing": None,
                "error": row_error,
            }
        )

    elapsed = time.perf_counter() - t0_total
    for cap in all_captions:
        cap["timing"] = {"seconds": float(elapsed)}

    out = batch_df.copy()
    out["vlm_caption"] = all_captions
    return out


def _caption_single(
    *,
    image_b64: str,
    prompt: str,
    system_prompt: Optional[str],
    model: Any,
    use_remote: bool,
    invoke_url: str,
    api_key: Optional[str],
    model_name: str,
    temperature: float,
) -> str:
    """Generate a single caption via local model or remote endpoint."""
    if use_remote:
        from nv_ingest_api.internal.transform.caption_image import _generate_captions

        captions = _generate_captions(
            base64_images=[image_b64],
            prompt=prompt,
            system_prompt=system_prompt,
            api_key=api_key or "",
            endpoint_url=invoke_url,
            model_name=model_name,
            temperature=temperature,
        )
        return captions[0] if captions else ""
    else:
        return model.caption_single(image_b64, prompt, system_prompt=system_prompt)


class VLMCaptioningActor:
    """
    Ray-friendly callable that initializes a VLM captioner once per actor.

    When ``invoke_url`` is provided, delegates to a remote vLLM / NIM
    HTTP endpoint.  Otherwise, loads ``NemotronNanoVL12BV2`` locally (requires
    GPU).

    This is a drop-in ``map_batches`` stage for the batch pipeline and is also
    usable directly via ``vlm_caption_images()`` in the inprocess pipeline.

    Follows the ``NemotronParseActor`` pattern (plain class with ``__slots__``
    and ``__call__``).
    """

    __slots__ = (
        "_model",
        "_invoke_url",
        "_api_key",
        "_request_timeout_s",
        "_prompt",
        "_system_prompt",
        "_model_name",
        "_max_tokens",
        "_temperature",
        "_caption_page_images",
        "_caption_tables",
        "_caption_charts",
        "_caption_infographics",
        "_remote_retry",
    )

    def __init__(
        self,
        *,
        invoke_url: Optional[str] = None,
        prompt: str = "Caption the content of this image:",
        system_prompt: Optional[str] = "/no_think",
        model_name: str = "nvidia/nemotron-nano-12b-v2-vl",
        max_tokens: int = 512,
        temperature: float = 1.0,
        caption_page_images: bool = True,
        caption_tables: bool = False,
        caption_charts: bool = False,
        caption_infographics: bool = False,
        api_key: Optional[str] = None,
        request_timeout_s: float = 120.0,
        device: Optional[str] = None,
        hf_cache_dir: Optional[str] = None,
        model_id: Optional[str] = None,
        remote_max_pool_workers: int = 16,
        remote_max_retries: int = 10,
        remote_max_429_retries: int = 5,
    ) -> None:
        self._invoke_url = (invoke_url or "").strip()
        self._api_key = api_key
        self._request_timeout_s = float(request_timeout_s)
        self._prompt = str(prompt)
        self._system_prompt = system_prompt
        self._model_name = str(model_name)
        self._max_tokens = int(max_tokens)
        self._temperature = float(temperature)
        self._caption_page_images = bool(caption_page_images)
        self._caption_tables = bool(caption_tables)
        self._caption_charts = bool(caption_charts)
        self._caption_infographics = bool(caption_infographics)
        self._remote_retry = RemoteRetryParams(
            remote_max_pool_workers=int(remote_max_pool_workers),
            remote_max_retries=int(remote_max_retries),
            remote_max_429_retries=int(remote_max_429_retries),
        )

        if self._invoke_url:
            self._model = None
        else:
            from nemo_retriever.model.local import NemotronNanoVL12BV2

            self._model = NemotronNanoVL12BV2(
                device=device,
                hf_cache_dir=hf_cache_dir,
                model_id=model_id,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )

    def __call__(self, batch_df: Any, **override_kwargs: Any) -> Any:
        try:
            return vlm_caption_images(
                batch_df,
                model=self._model,
                invoke_url=self._invoke_url,
                api_key=self._api_key,
                model_name=self._model_name,
                prompt=self._prompt,
                system_prompt=self._system_prompt,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                caption_page_images=self._caption_page_images,
                caption_tables=self._caption_tables,
                caption_charts=self._caption_charts,
                caption_infographics=self._caption_infographics,
                request_timeout_s=self._request_timeout_s,
                remote_retry=self._remote_retry,
                **override_kwargs,
            )
        except BaseException as e:
            if isinstance(batch_df, pd.DataFrame):
                out = batch_df.copy()
                n = len(out.index)
                payload = {
                    "page_caption": None,
                    "element_captions": [],
                    "timing": None,
                    "error": {
                        "stage": "vlm_captioning_actor_call",
                        "type": e.__class__.__name__,
                        "message": str(e),
                    },
                }
                out["vlm_caption"] = [payload for _ in range(n)]
                return out
            return [
                {
                    "vlm_caption": {
                        "page_caption": None,
                        "element_captions": [],
                        "timing": None,
                        "error": {
                            "stage": "vlm_captioning_actor_call",
                            "type": e.__class__.__name__,
                            "message": str(e),
                        },
                    }
                }
            ]
