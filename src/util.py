from pathlib import Path

import cv2
import numpy as np


def get_project_root() -> Path:
    """
    Get the root directory of the project.

    Returns:
        Path: The root directory of the project.
    """
    return Path(__file__).parent


def get_data_dir() -> Path:
    """
    Get the data directory path.

    Returns:
        Path: The data directory path.
    """
    return Path("/data")


def get_logs_path() -> Path:
    """
    Get the logs directory path.

    Returns:
        Path: The logs directory path.
    """
    return get_data_dir() / "logs"


def get_model_path() -> Path:
    """
    Get the path to the YOLO model weights file.

    Returns:
        Path: The path to the YOLO model weights file.
    """
    return get_project_root() / "model" / "weights" / "best.pt"


def get_output_path() -> Path:
    """
    Get the output directory path.

    Returns:
        Path: The output directory path.
    """
    return get_data_dir() / "output"


def get_input_path() -> Path:
    """
    Get the input directory path.

    Returns:
        Path: The input directory path.
    """
    return get_data_dir() / "input"


def get_test_input_path() -> Path:
    """
    Get the test input directory path.

    Returns:
        Path: The test input directory path.
    """
    return get_data_dir() / "test_input"


def xyxy_to_xywh(
    xyxy: tuple[float, float, float, float], width: int, height: int
) -> tuple[float, float, float, float]:
    """
    Convert bounding box from (x_min, y_min, x_max, y_max) format to (x_center, y_center, width, height) format, normalized by image width and height.

    Args:
        xyxy (tuple[float, float, float, float]): Bounding box in (x_min, y_min, x_max, y_max) format.
        width (int): Width of the image.
        height (int): Height of the image.

    Returns:
        tuple[float, float, float, float]: Bounding box in (x_center, y_center, width, height) format, normalized.
    """

    x_min, y_min, x_max, y_max = xyxy

    w = x_max - x_min
    h = y_max - y_min

    x_center = x_min + w / 2
    y_center = y_min + h / 2

    return (
        x_center / width,
        y_center / height,
        w / width,
        h / height,
    )


def xywh_to_xyxy(
    xywh: tuple[float, float, float, float], width: int, height: int
) -> tuple[float, float, float, float]:
    """
    Convert bounding box from (x_center, y_center, width, height) format to (x_min, y_min, x_max, y_max) format.

    Args:
        xywh (tuple[float, float, float, float]): Bounding box in (x_center, y_center, width, height) format.
        width (int): Width of the image.
        height (int): Height of the image.

    Returns:
        tuple[float, float, float, float]: Bounding box in (x_min, y_min, x_max, y_max) format.
    """

    x_center_norm, y_center_norm, w_norm, h_norm = xywh

    w = w_norm * width
    h = h_norm * height

    x_center = x_center_norm * width
    y_center = y_center_norm * height

    x_min = x_center - w / 2
    y_min = y_center - h / 2
    x_max = x_center + w / 2
    y_max = y_center + h / 2

    return (x_min, y_min, x_max, y_max)


def save_labels(
    boxes: list[tuple[float, float, float, float]], cls: int, path: Path
) -> None:
    """
    Save bounding box labels to a file in YOLO format.

    Args:
        boxes (list[tuple[float, float, float, float]]): List of bounding boxes in (x_center, y_center, width, height) format, normalized.
        cls (int): Class ID for all bounding boxes.
        path (Path): Path to the output label file.
    """
    with open(path, "w") as f:
        f.writelines(f"{cls} {x} {y} {w} {h}\n" for x, y, w, h in boxes)


def load_labels(path: Path) -> list[tuple[float, float, float, float]]:
    """
    Load bounding box labels from a file in YOLO format.

    Args:
        path (Path): Path to the label file.

    Returns:
        list[tuple[float, float, float, float]]: List of bounding boxes in (x_center, y_center, width, height) format, normalized.
    """
    with open(path) as f:
        boxes = []

        for line in f:
            if line.strip():
                values = line.split()[1:]  # Skip the class ID
                boxes.append(tuple(float(v) for v in values))

        return boxes


def draw_boxes(
    image_array: np.ndarray,
    boxes_xyxy: list[tuple[float, float, float, float]],
    thickness: int = 3,
    color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """
    Draw bounding boxes on an image.

    Args:
        image_array (np.ndarray): The image on which to draw the bounding boxes.
        boxes_xyxy (list[tuple[float, float, float, float]]): List of bounding boxes in (x_min, y_min, x_max, y_max) format.
        color (tuple[int, int, int], optional): Color of the bounding boxes in BGR format. Defaults to green (0, 255, 0).

    Returns:
        np.ndarray: The image with bounding boxes drawn on it.
    """

    for x_min, y_min, x_max, y_max in boxes_xyxy:
        cv2.rectangle(
            image_array,
            (int(x_min), int(y_min)),
            (int(x_max), int(y_max)),
            color=color,
            thickness=thickness,
        )

    return image_array


def del_dir_contents(folder_path: Path) -> None:
    """
    Delete all contents of a directory.

    Args:
        folder_path (Path): Path to the directory whose contents are to be deleted.
    """

    if not folder_path.is_dir():
        return

    for item in folder_path.iterdir():
        if item.is_dir():
            del_dir_contents(item)
            item.rmdir()
        else:
            item.unlink()
