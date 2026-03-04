# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
from PIL import Image


@dataclass
class NemotronNanoVL12BV2:
    """
    Local HuggingFace wrapper for ``nvidia/Nemotron-Nano-VL-12B-V2-FP8``.

    Loads the model via ``AutoModelForCausalLM`` + ``AutoProcessor`` +
    ``AutoTokenizer`` and runs image-to-text caption generation.

    Supports text, image, and text+image inputs via chat-template formatting.
    Images are processed one at a time (12B model is large; batching is handled
    at the outer level by the actor/pipeline).
    """

    device: Optional[str] = None
    hf_cache_dir: Optional[str] = None
    model_id: Optional[str] = None
    max_tokens: int = 512
    temperature: float = 1.0

    _model: Any = field(default=None, init=False, repr=False)
    _tokenizer: Any = field(default=None, init=False, repr=False)
    _processor: Any = field(default=None, init=False, repr=False)
    _device: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

        model_id = self.model_id or "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-FP8"
        dev = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        hf_cache_dir = self.hf_cache_dir or str(Path.home() / ".cache" / "huggingface")

        use_gpu = dev.type == "cuda"
        for attn_impl in ("flash_attention_2", "sdpa", "eager"):
            try:
                kwargs: dict[str, Any] = {
                    "trust_remote_code": True,
                    "torch_dtype": torch.bfloat16,
                    "attn_implementation": attn_impl,
                    "cache_dir": hf_cache_dir,
                }
                if attn_impl == "flash_attention_2" and use_gpu:
                    kwargs["device_map"] = dev
                self._model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
                break
            except (ValueError, ImportError):
                if attn_impl == "eager":
                    raise
                continue

        if not hasattr(self._model, "device_map"):
            self._model = self._model.to(dev)
        self._model.eval()
        self._device = dev

        self._tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=hf_cache_dir)
        self._processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True, cache_dir=hf_cache_dir
        )

    @staticmethod
    def _b64_to_pil(image_b64: str) -> Image.Image:
        """Decode a base64-encoded image string to a PIL Image."""
        raw = base64.b64decode(image_b64)
        return Image.open(io.BytesIO(raw)).convert("RGB")

    def caption(
        self,
        images_b64: Sequence[str],
        prompt: str,
        system_prompt: str | None = None,
    ) -> list[str]:
        """Generate captions for a list of base64-encoded images.

        Each image is processed individually (the 12B model is memory-intensive).

        Parameters
        ----------
        images_b64 : Sequence[str]
            Base64-encoded PNG/JPEG image strings.
        prompt : str
            User prompt for caption generation.
        system_prompt : str | None
            Optional system prompt (e.g. ``"/no_think"``).

        Returns
        -------
        list[str]
            One caption per input image.
        """
        captions: list[str] = []
        for b64 in images_b64:
            caption = self.caption_single(b64, prompt, system_prompt=system_prompt)
            captions.append(caption)
        return captions

    def caption_single(
        self,
        image_b64: str,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        """Generate a caption for a single base64-encoded image.

        Parameters
        ----------
        image_b64 : str
            Base64-encoded PNG/JPEG image string.
        prompt : str
            User prompt for caption generation.
        system_prompt : str | None
            Optional system prompt (e.g. ``"/no_think"``).

        Returns
        -------
        str
            Generated caption text.
        """
        pil_image = self._b64_to_pil(image_b64)

        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": ""},
                    {"type": "text", "text": prompt},
                ],
            }
        )

        text_prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text_prompt], images=[pil_image], return_tensors="pt"
        ).to(self._device)

        with torch.inference_mode():
            generated_ids = self._model.generate(
                pixel_values=inputs.pixel_values,
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=self.max_tokens,
                do_sample=self.temperature > 0,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        decoded = self._processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return decoded[0].strip() if decoded else ""
