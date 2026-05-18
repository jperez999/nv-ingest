# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mixin marking operators that need a post-execution sink callback."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class GraphSink(ABC):
    """Mixin marking operators that need a post-execution ``sink`` callback.

    Operators that inherit from both :class:`AbstractOperator` and this class
    will have their :meth:`sink` method invoked by the executor after the
    graph has finished executing, in graph-traversal order::

        class MySink(AbstractOperator, GraphSink):
            def sink(self, records, **kwargs):
                ...

    Executors detect sinks via ``isinstance(op, GraphSink)`` (in-process) or
    ``issubclass(node.operator_class, GraphSink)`` (Ray) and invoke
    ``op.sink(records, **kwargs)`` with the final dataset and the same
    keyword arguments that were passed to ``executor.ingest``.
    """

    @abstractmethod
    def sink(self, records: Any, **kwargs: Any) -> None:
        """Run the side-effect for this sink after graph execution.

        Parameters
        ----------
        records
            The final dataset produced by the graph (typically a
            ``pandas.DataFrame``).
        **kwargs
            Keyword arguments forwarded from ``executor.ingest``.
        """
        ...
