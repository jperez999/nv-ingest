"""Pending-work placement and reservation cleanup without CUDA or Ray workers."""

import asyncio
from types import SimpleNamespace

import pytest

from nemo_retriever.operators.extract.pdf.gpu_transport import (
    RendererPool,
    _cost_aware_scheduling_enabled,
    submit_scene_batch,
)


def pool_with_producers(cost_aware):
    pool = RendererPool("unused", rgb_pool_bytes=0, cost_aware=cost_aware)
    pool.producers = {("node", "gpu-a"): object(), ("node", "gpu-b"): object(), ("other", "gpu-c"): object()}
    return pool


def test_default_dispatch_keeps_same_node_round_robin():
    pool = pool_with_producers(False)
    picks = [asyncio.run(pool.get_next("node", 100)) for _ in range(3)]
    assert picks == [pool.producers[("node", "gpu-a")], pool.producers[("node", "gpu-b")], picks[0]]
    assert not pool._reservations


def test_heavy_batches_use_different_producers_until_work_finishes():
    pool = pool_with_producers(True)
    first = asyncio.run(pool.get_next("node", 2_000_000_000))
    second = asyncio.run(pool.get_next("node", 3_000_000_000))
    assert first["producer"] is not second["producer"]
    assert first["reservation"] == 0
    small = asyncio.run(pool.get_next("node", 10))
    assert small["producer"] is first["producer"]
    assert pool.release(first["reservation"])
    assert not pool.release(first["reservation"])
    next_work = asyncio.run(pool.get_next("node", 4_000_000_000))
    assert next_work["producer"] is first["producer"]
    for result in (second, small, next_work):
        assert pool.release(result["reservation"])
    assert not pool._reservations
    assert all(info["pending_scene_bytes"] == info["pending_batches"] == 0 for info in pool._dispatch.values())
    assert pool._dispatch[("node", "gpu-a")]["peak_pending_scene_bytes"] == 4_000_000_010
    assert all(info["pending_heavy_batches"] == 0 for info in pool._dispatch.values())


def test_normal_batches_still_feed_a_gpu_with_heavy_work():
    pool = pool_with_producers(True)
    heavy = asyncio.run(pool.get_next("node", 5_000_000_000))
    small_a = asyncio.run(pool.get_next("node", 10))
    small_b = asyncio.run(pool.get_next("node", 10))
    assert small_a["producer"] is not heavy["producer"]
    assert small_b["producer"] is heavy["producer"]
    next_heavy = asyncio.run(pool.get_next("node", 2_000_000_000))
    assert next_heavy["producer"] is not heavy["producer"]
    for work in (heavy, small_a, small_b, next_heavy):
        pool.release(work["reservation"])
    assert all(info["pending_heavy_batches"] == 0 for info in pool._dispatch.values())


@pytest.mark.parametrize("raw,expected", [(None, False), ("0", False), ("1", True)])
def test_cost_setting_is_explicit_and_default_off(raw, expected, monkeypatch):
    if raw is None:
        monkeypatch.delenv("CUDA_PDF_COST_AWARE_SCHEDULING", raising=False)
    else:
        monkeypatch.setenv("CUDA_PDF_COST_AWARE_SCHEDULING", raw)
    assert _cost_aware_scheduling_enabled() is expected


@pytest.mark.parametrize("raw", ["", "true", "-1", "2"])
def test_invalid_cost_setting_fails_setup(raw, monkeypatch):
    monkeypatch.setenv("CUDA_PDF_COST_AWARE_SCHEDULING", raw)
    with pytest.raises(ValueError, match="CUDA_PDF_COST_AWARE_SCHEDULING"):
        _cost_aware_scheduling_enabled()


def test_empty_scene_batch_needs_no_renderer():
    assert submit_scene_batch([], None) == []


def test_failed_replay_releases_reservation_zero(monkeypatch):
    import ray

    released = []
    failure = RuntimeError("render failed")
    producer = SimpleNamespace(render_handles=SimpleNamespace(remote=lambda handles: failure))
    pool = SimpleNamespace(
        get_next=SimpleNamespace(remote=lambda node, cost: {"producer": producer, "reservation": 0}),
        release=SimpleNamespace(remote=lambda token: released.append(token)),
    )

    def get(value):
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(ray, "get", get)
    monkeypatch.setattr(ray, "get_runtime_context", lambda: SimpleNamespace(get_node_id=lambda: "node"))
    scene = SimpleNamespace(handle=lambda: SimpleNamespace(size=100))
    with pytest.raises(RuntimeError, match="render failed"):
        submit_scene_batch([scene], pool)
    assert released == [0]
