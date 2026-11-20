# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from nemo_retriever.models.nim.primitives.model_interface import yolox


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("count", [0, 1, 2, 7, 64, 256, 257])
@pytest.mark.parametrize("threshold", [0.0, 0.01, 0.5, 1.0])
def test_bounded_matches_keep_original_indices(dtype, count, threshold):
    rng = np.random.default_rng(20261120)
    boxes = np.zeros((count, 8), dtype=dtype)
    xy = rng.random((count, 4)).astype(dtype)
    xy[:, 2:] += xy[:, :2]
    boxes[:, 4:] = xy

    matches = yolox._matching_boxes_bounded(boxes, threshold)
    if count > 256:
        assert matches is None
        return

    for index in range(count):
        others = [i for i in range(count) if i != index]
        relative, _ = yolox.find_matching_box_fast(boxes[others], boxes[index], threshold)
        expected = others[relative] if relative != -1 else -1
        assert matches[index] == expected


def test_bounded_matches_exclude_self_and_keep_first_tie():
    boxes = np.tile([0.0, 0.9, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0], (3, 1))

    assert yolox._matching_boxes_bounded(boxes, 0.01).tolist() == [1, 0, 0]
    assert yolox._matching_boxes_bounded(boxes, 1.0).tolist() == [-1, -1, -1]


@pytest.mark.parametrize("dtype", [np.float16, np.int64])
def test_bounded_matches_leave_other_dtypes_on_original_path(dtype):
    assert yolox._matching_boxes_bounded(np.zeros((2, 8), dtype=dtype), 0.01) is None


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_bounded_matches_leave_nonfinite_inputs_on_original_path(value):
    boxes = np.zeros((2, 8))
    boxes[0, 4] = value
    assert yolox._matching_boxes_bounded(boxes, 0.01) is None


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize("confidence", ["max", "avg"])
def test_singleton_fusion_keeps_float32_rounding_and_model_index(dtype, confidence):
    boxes = np.array(
        [[3, 0.75 + 2**-26, 1, 7, 0.125 + 2**-28, 0.25 + 2**-27, 0.75 + 2**-26, 0.875 + 2**-26]],
        dtype=dtype,
    )

    result = yolox.get_biggest_box(boxes, confidence)

    expected = np.array([3, 0.75, 1, -1, 0.125, 0.25, 0.75, 0.875], dtype=np.float32)
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == np.float32
    assert not np.shares_memory(result, boxes)


@pytest.mark.parametrize("merge", ["biggest", "weighted"])
@pytest.mark.parametrize("confidence", ["max", "avg"])
@pytest.mark.parametrize("class_agnostic", [False, True])
@pytest.mark.parametrize("count", [1, 10, 64, 257])
def test_fusion_cache_preserves_original_clustering_and_error_contract(
    monkeypatch, merge, confidence, class_agnostic, count
):
    rng = np.random.default_rng(20261120)
    boxes = rng.random((count, 4))
    boxes[:, 2:] = np.minimum(1, boxes[:, :2] + boxes[:, 2:] * 0.4)
    scores = rng.choice([0.1, 0.5, 0.9], size=count)
    labels = rng.integers(0, 6, size=count)

    def invoke():
        try:
            return yolox.weighted_boxes_fusion(
                [boxes],
                [scores],
                [labels],
                merge_type=merge,
                conf_type=confidence,
                class_agnostic=class_agnostic,
                iou_thr=0.01,
            )
        except Exception as error:
            return type(error), str(error)

    with monkeypatch.context() as reference:
        reference.setattr(yolox, "_matching_boxes_bounded", lambda *_: None)
        expected = invoke()
    result = invoke()

    if isinstance(expected[0], type):
        assert result == expected
    else:
        for actual, original in zip(result, expected):
            assert actual.dtype == original.dtype
            np.testing.assert_array_equal(actual, original)
