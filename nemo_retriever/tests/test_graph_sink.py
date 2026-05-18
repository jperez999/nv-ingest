# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the GraphSink mixin and executor integration."""

from __future__ import annotations

from typing import Any, ClassVar

import pandas as pd
import pytest

from nemo_retriever.graph import GraphSink
from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.graph.executor import InprocessExecutor
from nemo_retriever.graph.pipeline_graph import Graph


class _PassThroughOperator(AbstractOperator):
    """Operator that returns data untouched (no side effects)."""

    def __init__(self) -> None:
        super().__init__()

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> Any:
        return data

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data


class _RecordingSinkOperator(AbstractOperator, GraphSink):
    """Sink-flagged operator that records every sink call on a class variable.

    The executor reconstructs operators from ``operator_class(**operator_kwargs)``
    each run, so tests inspect this class-level list rather than instance state.
    """

    sink_calls: ClassVar[list[tuple[str, Any, dict[str, Any]]]] = []

    def __init__(self, label: str = "sink") -> None:
        super().__init__()
        self.label = label

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> Any:
        return data

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def sink(self, records: Any, **kwargs: Any) -> None:
        type(self).sink_calls.append((self.label, records, dict(kwargs)))


@pytest.fixture(autouse=True)
def _reset_sink_calls() -> None:
    _RecordingSinkOperator.sink_calls = []


def test_graph_sink_is_abstract() -> None:
    """GraphSink is an ABC and cannot be instantiated without `sink`."""

    class IncompleteOperator(AbstractOperator, GraphSink):
        def preprocess(self, data: Any, **kwargs: Any) -> Any:
            return data

        def process(self, data: Any, **kwargs: Any) -> Any:
            return data

        def postprocess(self, data: Any, **kwargs: Any) -> Any:
            return data

    with pytest.raises(TypeError):
        IncompleteOperator()  # type: ignore[abstract]


def test_sink_subclass_passes_isinstance_check() -> None:
    op = _RecordingSinkOperator()
    assert isinstance(op, GraphSink)
    assert issubclass(_RecordingSinkOperator, GraphSink)


def test_non_sink_operator_is_not_detected_as_graph_sink() -> None:
    op = _PassThroughOperator()
    assert not isinstance(op, GraphSink)


def test_inprocess_executor_invokes_sink_after_graph_completion() -> None:
    """The sink fires once after graph execution with the final DataFrame."""
    graph = _PassThroughOperator() >> _RecordingSinkOperator()
    executor = InprocessExecutor(graph, show_progress=False)

    df = pd.DataFrame({"value": [1, 2, 3]})
    result = executor.ingest(df)

    assert result is df
    assert len(_RecordingSinkOperator.sink_calls) == 1
    _label, received_records, received_kwargs = _RecordingSinkOperator.sink_calls[0]
    assert received_records is df
    assert received_kwargs == {}


def test_inprocess_executor_forwards_ingest_kwargs_to_sink() -> None:
    graph = _PassThroughOperator() >> _RecordingSinkOperator()
    executor = InprocessExecutor(graph, show_progress=False)

    df = pd.DataFrame({"value": [1]})
    executor.ingest(df, batch_id="abc", extra=42)

    assert len(_RecordingSinkOperator.sink_calls) == 1
    _label, _records, received_kwargs = _RecordingSinkOperator.sink_calls[0]
    assert received_kwargs == {"batch_id": "abc", "extra": 42}


def test_multiple_sinks_fire_in_graph_traversal_order() -> None:
    graph = _RecordingSinkOperator(label="first") >> _RecordingSinkOperator(label="second")
    executor = InprocessExecutor(graph, show_progress=False)

    executor.ingest(pd.DataFrame({"value": [1]}))

    labels = [call[0] for call in _RecordingSinkOperator.sink_calls]
    assert labels == ["first", "second"]


def test_graph_with_no_sinks_executes_without_invoking_sink_logic() -> None:
    """Regression: graphs without GraphSink operators behave as before."""
    graph = _PassThroughOperator() >> _PassThroughOperator()
    executor = InprocessExecutor(graph, show_progress=False)
    df = pd.DataFrame({"value": [1, 2]})

    result = executor.ingest(df)

    assert result is df
    assert _RecordingSinkOperator.sink_calls == []


def test_sink_runs_after_process_for_same_operator() -> None:
    """`process` and `sink` should both fire, in that order, on a sink operator."""

    class StageAndSink(AbstractOperator, GraphSink):
        events: ClassVar[list[str]] = []

        def preprocess(self, data: Any, **kwargs: Any) -> Any:
            return data

        def process(self, data: Any, **kwargs: Any) -> Any:
            type(self).events.append("process")
            return data

        def postprocess(self, data: Any, **kwargs: Any) -> Any:
            return data

        def sink(self, records: Any, **kwargs: Any) -> None:
            type(self).events.append("sink")

    graph = Graph()
    graph.add_root(StageAndSink())
    executor = InprocessExecutor(graph, show_progress=False)

    executor.ingest(pd.DataFrame({"value": [1]}))

    assert StageAndSink.events == ["process", "sink"]


# ---------------------------------------------------------------------------
# IngestVdbOperator integration with the GraphSink mixin
# ---------------------------------------------------------------------------


class _FakeVDB:
    """Minimal stand-in for the VDB interface used by IngestVdbOperator.

    The executor reconstructs the operator (and therefore its VDB) when it
    runs the graph, so tests that exercise the full executor path use the
    class-level lists below instead of instance state.
    """

    run_calls: ClassVar[list[Any]] = []
    sink_calls: ClassVar[list[tuple[Any, dict[str, Any]]]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def create_index(self, **kwargs: Any) -> None:
        return None

    def write_to_index(self, records: list, **kwargs: Any) -> None:
        return None

    def retrieval(self, vectors: list, **kwargs: Any) -> list:
        return []

    def run(self, records: Any) -> Any:
        type(self).run_calls.append(records)
        return records

    def sink(self, records: Any, **kwargs: Any) -> None:
        type(self).sink_calls.append((records, dict(kwargs)))


@pytest.fixture(autouse=True)
def _reset_fake_vdb_calls() -> None:
    _FakeVDB.run_calls = []
    _FakeVDB.sink_calls = []


def _vdb_graph_rows() -> list[dict[str, Any]]:
    return [
        {
            "text": "chunk",
            "text_embeddings_1b_v2": {"embedding": [0.1] * 2048},
            "path": "/tmp/doc.pdf",
            "page_number": 1,
            "metadata": {"content_metadata": {"type": "text"}},
        }
    ]


def test_ingest_vdb_operator_inherits_graph_sink() -> None:
    from nemo_retriever.vdb import IngestVdbOperator

    op = IngestVdbOperator(vdb=_FakeVDB())
    assert isinstance(op, GraphSink)


def test_ingest_vdb_operator_sink_does_all_vdb_work() -> None:
    """`sink` performs conversion + sidecar + delegates to VDB.sink."""
    from nemo_retriever.vdb import IngestVdbOperator

    op = IngestVdbOperator(vdb=_FakeVDB())
    rows = _vdb_graph_rows()

    op.sink(rows, batch_id="run-1")

    assert len(_FakeVDB.sink_calls) == 1
    converted_records, sink_kwargs = _FakeVDB.sink_calls[0]
    # `sink` converts graph rows into the nested NV-Ingest client VDB shape
    # before delegating to the underlying VDB's sink.
    assert converted_records[0][0]["document_type"] == "text"
    assert converted_records[0][0]["metadata"]["content"] == "chunk"
    assert sink_kwargs == {"batch_id": "run-1"}


def test_ingest_vdb_operator_process_is_a_noop() -> None:
    """`process` does no VDB work; all writes are deferred to `sink`."""
    from nemo_retriever.vdb import IngestVdbOperator

    op = IngestVdbOperator(vdb=_FakeVDB())
    rows = _vdb_graph_rows()

    assert op.process(rows) is rows

    assert _FakeVDB.run_calls == []
    assert _FakeVDB.sink_calls == []


def test_inprocess_executor_fires_ingest_vdb_operator_sink() -> None:
    """End-to-end: process is a no-op; the executor fires VDB.sink once."""
    from nemo_retriever.vdb import IngestVdbOperator

    graph = Graph()
    graph.add_root(IngestVdbOperator(vdb=_FakeVDB()))
    executor = InprocessExecutor(graph, show_progress=False)

    df = pd.DataFrame(_vdb_graph_rows())
    executor.ingest(df, request_id="req-9")

    # process is a pure pass-through — no per-batch VDB.run calls.
    assert _FakeVDB.run_calls == []
    # sink fires exactly once after graph execution, with kwargs forwarded.
    assert len(_FakeVDB.sink_calls) == 1
    _records, sink_kwargs = _FakeVDB.sink_calls[0]
    assert sink_kwargs == {"request_id": "req-9"}
