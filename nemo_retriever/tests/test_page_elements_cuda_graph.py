# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Cached decoder geometry and captured inference preserve values and ownership."""

import json

import pytest

torch = pytest.importorskip("torch")

from nemo_retriever.models.local.page_elements_cuda import _CachedGridHead, _CudaGraphBackbone  # noqa: E402
from nemotron_page_elements_v3.yolox.yolo_head import YOLOXHead  # noqa: E402


class DecoderHead(torch.nn.Module):
    decode_outputs = YOLOXHead.decode_outputs

    def __init__(self):
        super().__init__()
        self.decode_in_inference = True
        self.strides = [8]
        self.hw = []

    def forward(self, features):
        x = features[0]
        self.hw = [tuple(x.shape[-2:])]
        count = x.shape[-2] * x.shape[-1]
        raw = (
            torch.arange(x.shape[0] * count * 11, device=x.device, dtype=x.dtype).reshape(x.shape[0], count, 11) / 1000
        )
        return self.decode_outputs(raw, dtype=x.type()) if self.decode_in_inference else raw


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_grid_cache_preserves_geometry_changes_and_original_head(dtype):
    original = DecoderHead().eval()
    cached = _CachedGridHead(original).eval()
    for batch, height, width in [(8, 2, 3), (1, 2, 3), (8, 3, 3), (3, 2, 3), (8, 2, 3)]:
        features = [torch.zeros((batch, 4, height, width), dtype=dtype)]
        expected = original(features)
        actual = cached(features)
        assert expected.dtype == actual.dtype
        assert torch.equal(expected, actual)
        assert original.decode_in_inference is True
        assert cached.raw_head.decode_in_inference is False
        assert cached.builds == 1


@pytest.fixture
def cuda_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for captured detector execution")
    return torch.device("cuda")


def test_graph_replay_partial_batches_and_previous_results(cuda_device, monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_PDF_RDT_AUDIT_PATH", str(tmp_path / "transport.json"))
    layer = torch.nn.Sequential(torch.nn.Linear(4, 5), torch.nn.ReLU()).to(cuda_device).eval()
    captured = _CudaGraphBackbone(layer)
    x = torch.arange(32, device=cuda_device, dtype=torch.float32).reshape(8, 4) / 17
    with torch.inference_mode():
        old = captured(x)
        snapshot = old.clone()
        for next_input in [x + 1, x[:1], x[:3], x.t().contiguous().t(), x]:
            expected = layer(next_input)
            actual = captured(next_input)
            assert torch.equal(expected, actual)
            assert torch.equal(old, snapshot)
    assert captured.captures == 1
    assert captured.replays == 3
    assert captured.eager_calls == 3
    audit = json.loads(next(tmp_path.glob("transport.detector-*.json")).read_text())
    assert audit["captures"] == 1 and audit["replays"] == 1
    assert audit["counters_may_precede_final_batch"] is True


def test_autocast_weights_survive_context_exit_and_in_place_update(cuda_device, monkeypatch):
    monkeypatch.delenv("CUDA_PDF_RDT_AUDIT_PATH", raising=False)
    layer = torch.nn.Linear(4, 5).to(cuda_device).eval()
    captured = _CudaGraphBackbone(layer)
    x = torch.arange(32, device=cuda_device, dtype=torch.float32).reshape(8, 4) / 17
    with torch.inference_mode():
        with torch.autocast("cuda"):
            first = captured(x)
        layer.weight.add_(0.125)
        with torch.autocast("cuda"):
            expected = layer(x)
        with torch.autocast("cuda"):
            actual = captured(x)
        assert torch.equal(actual, expected)
        assert not torch.equal(first, actual)
    assert captured.captures == 1 and captured.replays == 2


def test_grad_enabled_execution_stays_eager(cuda_device):
    layer = torch.nn.Linear(4, 5).to(cuda_device).eval()
    captured = _CudaGraphBackbone(layer)
    result = captured(torch.ones((8, 4), device=cuda_device))
    assert result.requires_grad
    assert captured.graph is None
