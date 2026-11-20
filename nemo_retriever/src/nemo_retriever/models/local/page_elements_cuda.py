# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""GPU-input detector execution with resident decoding grids and CUDA Graphs."""

from __future__ import annotations

import copy
import json
import logging
import os
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


class _CachedGridHead(torch.nn.Module):
    def __init__(self, head):
        super().__init__()
        # Share immutable inference weights, but keep the original head's
        # decode flag and per-forward geometry attributes independent.
        self.raw_head = copy.copy(head)
        self.raw_head.decode_in_inference = False
        self.key = None
        self.grids = None
        self.strides = None
        self.builds = 0

    def forward(self, features):
        from nemotron_page_elements_v3.yolox.yolo_head import meshgrid

        raw = self.raw_head(features)
        key = (tuple(tuple(shape) for shape in self.raw_head.hw), features[0].dtype, features[0].device)
        if self.key is not None and key != self.key:
            return self.raw_head.decode_outputs(raw, dtype=features[0].type())
        if self.key is None:
            grids, strides = [], []
            for (height, width), stride in zip(self.raw_head.hw, self.raw_head.strides):
                y, x = meshgrid([torch.arange(height), torch.arange(width)])
                grid = torch.stack((x, y), 2).view(1, -1, 2)
                grids.append(grid)
                strides.append(torch.full((*grid.shape[:2], 1), stride))
            self.grids = torch.cat(grids, dim=1).type(features[0].type())
            self.strides = torch.cat(strides, dim=1).type(features[0].type())
            self.key = key
            self.builds += 1
        return torch.cat(
            (
                (raw[..., :2] + self.grids) * self.strides,
                torch.exp(raw[..., 2:4]) * self.strides,
                raw[..., 4:],
            ),
            dim=-1,
        )


class _CudaGraphBackbone(torch.nn.Module):
    """Capture one fixed inference signature; other inputs stay eager on GPU."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.key = None
        self.graph = None
        self.static_input = None
        self.static_output = None
        self.grid_owners = ()
        self.captures = 0
        self.replays = 0
        self.eager_calls = 0

    @staticmethod
    def _signature(x):
        return (
            tuple(x.shape),
            tuple(x.stride()),
            x.dtype,
            x.device,
            torch.is_autocast_enabled("cuda"),
            torch.get_autocast_dtype("cuda"),
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32,
            torch.backends.cudnn.deterministic,
        )

    def _capture(self, x, key):
        self.key = key
        self.static_input = torch.empty_like(x)
        self.static_input.copy_(x)
        current = torch.cuda.current_stream(x.device)
        stream = torch.cuda.Stream(device=x.device)
        stream.wait_stream(current)
        # Weight casts must belong to the graph. A surrounding autocast cache
        # can otherwise release a captured FP16 weight pointer after this call.
        with torch.cuda.stream(stream), torch.autocast("cuda", enabled=key[4], dtype=key[5], cache_enabled=False):
            for _ in range(3):
                self.backbone(self.static_input)
        current.wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream), torch.autocast(
            "cuda", enabled=key[4], dtype=key[5], cache_enabled=False
        ):
            self.static_output = self.backbone(self.static_input)
        head = getattr(self.backbone, "head", None)
        if isinstance(head, _CachedGridHead):
            self.grid_owners = (head.grids, head.strides)
        current.wait_stream(stream)
        self.captures += 1

    def _audit(self, x):
        configured = os.environ.get("CUDA_PDF_RDT_AUDIT_PATH")
        if not configured or not x.is_cuda:
            return
        path = Path(configured)
        path = path.with_name(f"{path.stem}.detector-{os.getpid()}.json")
        head = getattr(self.backbone, "head", None)
        data = {
            "pid": os.getpid(),
            "gpu_uuid": str(torch.cuda.get_device_properties(x.device).uuid),
            "captures": self.captures,
            "replays": self.replays,
            "eager_calls": self.eager_calls,
            "grid_builds": getattr(head, "builds", 0),
            "counters_may_precede_final_batch": True,
        }
        try:
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(data) + "\n")
            temporary.replace(path)
        except OSError:
            logger.debug("Could not write optional detector execution audit", exc_info=True)

    def forward(self, x):
        if not x.is_cuda or x.shape[0] != 8 or self.backbone.training or torch.is_grad_enabled():
            self.eager_calls += 1
            return self.backbone(x)
        key = self._signature(x)
        if self.graph is not None and key != self.key:
            self.eager_calls += 1
            return self.backbone(x)
        with torch.cuda.device(x.device):
            if self.graph is None:
                self._capture(x, key)
            self.static_input.copy_(x)
            self.graph.replay()
            self.replays += 1
            # Upstream NMS modifies its input; return independent storage.
            result = self.static_output.clone()
            if self.replays == 1 or self.replays % 512 == 0:
                self._audit(x)
            return result


def make_cuda_detector(model):
    from nemotron_page_elements_v3.model import YoloXWrapper
    from nemotron_page_elements_v3.yolox.yolox import YOLOX

    if not isinstance(model, YoloXWrapper) or not isinstance(model.model, YOLOX):
        return model
    if not model.model.head.decode_in_inference:
        return model
    network = YOLOX(model.model.backbone, _CachedGridHead(model.model.head)).eval()
    return YoloXWrapper(_CudaGraphBackbone(network), model.config).eval()
