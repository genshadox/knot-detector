import pytest

from test import compare_labels


@pytest.mark.parametrize(
    ("ground_truth", "predictions", "expected"),
    [
        pytest.param([], [], (0, 0, 0, []), id="no-boxes"),
        pytest.param(
            [(0.0, 0.0, 10.0, 10.0)], [], (0, 1, 0, []), id="missing-prediction"
        ),
        pytest.param(
            [], [(0.0, 0.0, 10.0, 10.0)], (0, 0, 1, []), id="unexpected-prediction"
        ),
    ],
)
def test_compare_labels_handles_empty_inputs(ground_truth, predictions, expected):
    """
    Test that compare_labels correctly handles empty input lists.

    Parameters:
    - ground_truth: List of ground truth bounding boxes.
    - predictions: List of predicted bounding boxes.
    - expected: Expected result tuple (matched, missed, false_positives, matched_ious).
    """

    assert compare_labels(ground_truth, predictions) == expected


def test_compare_labels_counts_exact_matches():
    """
    Test that compare_labels correctly counts exact matches between ground truth and predicted boxes.
    """

    boxes = [
        (0.0, 0.0, 10.0, 10.0),
        (20.0, 20.0, 30.0, 30.0),
    ]

    matched, missed, false_positives, matched_ious = compare_labels(boxes, boxes)

    assert (matched, missed, false_positives) == (2, 0, 0)
    assert matched_ious == pytest.approx([1.0, 1.0])


def test_compare_labels_rejects_iou_below_threshold():
    """
    Test that compare_labels correctly rejects matches with IoU below the specified threshold.
    """

    ground_truth = [(0.0, 0.0, 10.0, 10.0)]
    predictions = [(6.0, 0.0, 16.0, 10.0)]

    result = compare_labels(ground_truth, predictions, iou_threshold=0.5)

    assert result == (0, 1, 1, [])


def test_compare_labels_does_not_double_count_predictions_for_one_box():
    """
    Test that compare_labels does not double count predictions for a single ground truth box.
    """

    ground_truth = [(0.0, 0.0, 10.0, 10.0)]
    predictions = [
        (0.0, 0.0, 10.0, 10.0),
        (1.0, 1.0, 9.0, 9.0),
    ]

    matched, missed, false_positives, matched_ious = compare_labels(
        ground_truth, predictions
    )

    assert (matched, missed, false_positives) == (1, 0, 1)
    assert matched_ious == pytest.approx([1.0])
