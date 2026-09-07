import json
import logging
from pathlib import Path

import cv2
import torch
from scipy.optimize import linear_sum_assignment
from ultralytics.utils.metrics import box_iou

from params import TEST_IOU_THRESHOLD
from util import draw_boxes, load_labels, xywh_to_xyxy

logger = logging.getLogger(__name__)


def load_labels_xyxy(
    path: Path, width: int, height: int
) -> list[tuple[float, float, float, float]]:
    """
    Load bounding box labels from a file in xywh format and convert them to (x_min, y_min, x_max, y_max) format.

    Args:
        path (Path): Path to the label file.
        width (int): Width of the image.
        height (int): Height of the image.

    Returns:
        list[tuple[float, float, float, float]]: List of bounding boxes in (x_min, y_min, x_max, y_max) format.
    """

    boxes_xywh = load_labels(path)
    boxes_xyxy = [xywh_to_xyxy(box, width, height) for box in boxes_xywh]

    return boxes_xyxy


def compare_labels(
    gt_boxes_xyxy: list[tuple[float, float, float, float]],
    pred_boxes_xyxy: list[tuple[float, float, float, float]],
    iou_threshold: float = 0.5,
) -> tuple[int, int, int, list[float]]:
    """
    Compare ground truth bounding boxes with predicted bounding boxes and calculate metrics.

    Args:
        gt_boxes_xyxy (list[tuple[float, float, float, float]]): List of ground truth bounding boxes in (x_min, y_min, x_max, y_max)
        pred_boxes_xyxy (list[tuple[float, float, float, float]]): List of predicted bounding boxes in (x_min, y_min, x_max, y_max)
        iou_threshold (float): IoU threshold to consider a match between ground truth and predicted boxes. Default is 0.5.

    Returns:
        tuple[int, int, int, list[float]]: A tuple containing the number of matched boxes, missed boxes, false positives, and a list of IoUs of matched boxes.
    """

    if not gt_boxes_xyxy or not pred_boxes_xyxy:
        return 0, len(gt_boxes_xyxy or []), len(pred_boxes_xyxy or []), []

    # This constructs a matrix of IoU values between each ground truth box and each predicted box.
    # If, for instance, we have 100% match, this would produce identity matrix. If we have 0% match, this would produce a matrix of zeros.
    iou_matrix = box_iou(torch.tensor(gt_boxes_xyxy), torch.tensor(pred_boxes_xyxy))

    # Find the optimal assignment of ground truth boxes to predicted boxes using the Hungarian algorithm(https://en.wikipedia.org/wiki/Hungarian_algorithm)
    # We do this to eliminate the possibility of double counting matches.
    # For example, if we have 1 ground truth box and 2 predicted boxes that both match it, we only want to count 1 match.
    gt_idx, pred_idx = linear_sum_assignment(-iou_matrix.numpy())

    matched_ious = []
    for g, p in zip(gt_idx, pred_idx):
        iou = iou_matrix[g, p].item()

        if iou >= iou_threshold:
            matched_ious.append(iou)

    matched = len(matched_ious)

    return (
        matched,
        len(gt_boxes_xyxy) - matched,
        len(pred_boxes_xyxy) - matched,
        matched_ious,
    )


def summarize(
    matched: int, missed: int, false_positives: int, matched_ious: list[float]
) -> dict:
    """
    Summarize the comparison metrics into a dictionary.

    Args:
        matched (int): Number of matched boxes.
        missed (int): Number of missed boxes.
        false_positives (int): Number of false positive boxes.
        matched_ious (list[float]): List of IoUs of matched boxes.
    Returns:
        dict: A dictionary containing the number of matched boxes, missed boxes, false positives, mean IoU of matched boxes, precision, and recall.
    """

    precision = (
        matched / (matched + false_positives) if (matched + false_positives) else 0.0
    )
    recall = matched / (matched + missed) if (matched + missed) else 0.0

    return {
        "matched": matched,
        "missed": missed,
        "false_positives": false_positives,
        "mean_iou_matched": sum(matched_ious) / matched if matched else 0.0,
        "precision": precision,
        "recall": recall,
    }


def test_results(test_path: Path, compare_path: Path):
    """
    Test the inference results by comparing the output files with the expected results and save the comparison metrics to JSON files.

    Args:
        test_path (Path): Path to the folder containing the input board images and their corresponding label files.
        compare_path (Path): Path to the folder containing the output files to compare.

    """

    output_board_files = list(compare_path.glob("*.png"))

    total_matched = 0
    total_missed = 0
    total_false_positives = 0
    total_matched_ious = []

    for output_board_file in output_board_files:
        board_id = output_board_file.stem
        test_label_file = test_path / f"{board_id}.txt"

        if not test_label_file.exists():
            logger.warning(
                f"Input label file {test_label_file} does not exist. Skipping comparison for board {board_id}."
            )
            continue

        output_label_file = compare_path / f"{board_id}.txt"

        if not output_label_file.exists():
            logger.warning(
                f"Output label file {output_label_file} does not exist. Skipping comparison for board {board_id}."
            )
            continue

        board_img_array = cv2.imread(str(output_board_file))
        height, width = board_img_array.shape[:2]

        gt_boxes_xyxy = load_labels_xyxy(test_label_file, width, height)

        board_img_array = draw_boxes(
            board_img_array, gt_boxes_xyxy, thickness=2, color=(255, 0, 0)
        )

        cv2.imwrite(str(output_board_file), board_img_array)

        output_boxes_xyxy = load_labels_xyxy(output_label_file, width, height)

        matched, missed, false_positives, matched_ious = compare_labels(
            gt_boxes_xyxy,
            output_boxes_xyxy,
            iou_threshold=TEST_IOU_THRESHOLD,
        )

        total_matched += matched
        total_missed += missed
        total_false_positives += false_positives
        total_matched_ious.extend(matched_ious)

        with open(compare_path / f"{board_id}_compare.json", "w") as file:
            json.dump(
                summarize(matched, missed, false_positives, matched_ious),
                file,
                indent=2,
            )

    with open(compare_path / "summary.json", "w") as file:
        json.dump(
            summarize(
                total_matched, total_missed, total_false_positives, total_matched_ious
            ),
            file,
            indent=2,
        )
