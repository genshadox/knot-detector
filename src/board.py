import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics.engine.results import Boxes

from util import draw_boxes, save_labels, xyxy_to_xywh

logger = logging.getLogger(__name__)


def build_boards(paths: list[Path]) -> Iterator["Board"]:
    """
    Build Board objects from a list of image paths.

    Args:
        paths (list[Path]): List of image file paths.

    Returns:
        Iterator[Board]: Iterator of Board objects with associated frames.
    """

    boards: dict[str, list[Frame]] = {}

    for path in paths:
        board_id, frame_num = path.stem.split("_")
        boards.setdefault(board_id, []).append(Frame(path, int(frame_num)))

    for board_id, frames in boards.items():
        frames.sort(key=lambda f: f.frame_num)
        yield Board(board_id, frames)


@dataclass
class Frame:
    """
    Represents a single frame in a board.
    """

    path: Path
    frame_num: int


@dataclass
class Window:
    """
    Represents a window within a board, which is a subset of frames.
    """

    x_offset: int
    width: int
    boxes: Boxes


class Board:
    """
    Represents a board consisting of multiple frames.
    """

    def __init__(self, id: str, frames: list[Frame]):
        """
        Initialize a Board object.

        Args:
            id (str): Board ID.
            frames (list[Frame]): List of Frame objects associated with the board.
        """

        self.id: str = id
        self.frames: list[Frame] = frames
        self.CLS: int = 0  # we have only class 0

        images = []

        for frame in self.frames:
            image = cv2.imread(str(frame.path))

            if image is None:
                raise ValueError(f"Failed to read frame image: {frame.path}")

            images.append(image)

        self.image: np.ndarray = np.hstack(images)  # same height assumed across frames

    def get_window_data(self, x: int, width: int) -> np.ndarray:

        return self.image[:, x : x + width]

    def add_window(self, x: int, width: int, boxes: Boxes):
        """
        Add a window's results to the board.

        Args:
            x (int): The x-coordinate of the window's top-left corner in the board image.
            width (int): The width of the window.
            boxes (Boxes): Bounding boxes and confidence scores for the window.
        """

        if not hasattr(self, "windows"):
            self.windows: list[Window] = []

        self.windows.append(Window(x_offset=x, width=width, boxes=boxes))

    def save_results(self, output_folder: Path):
        """
        Save the results of the board to the specified output folder.

        Args:
            output_folder (Path): Path to the folder where the results will be saved.
        """

        boxes_xyxy, boxes_xywh = self.to_board_space()

        save_labels(boxes_xywh, self.CLS, output_folder / f"{self.id}.txt")

        self.plot_board(boxes_xyxy, output_folder / f"{self.id}.png")

    def plot_board(self, boxes_xyxy: list[tuple], output_path: Path):
        """
        Plot the board by stitching frames together and drawing bounding boxes.

        Args:
            boxes_xyxy (list[tuple]): List of bounding boxes in xyxy format.
            output_path (Path): Path to save the plotted board image.
        """

        data = draw_boxes(self.image, boxes_xyxy, thickness=3, color=(0, 255, 0))

        cv2.imwrite(str(output_path), data)

    def to_board_space(self) -> tuple[list, list]:
        """
            Convert window-level bounding boxes to board-level coordinates and merge overlapping boxes across windows.

        Returns:
            tuple[list, list]: A tuple containing a list of bounding boxes in xyxy format and a list of bounding boxes in xywh format.
        """

        boxes_xyxy_conf = []

        for window in self.windows:
            assert window.boxes is not None

            for box, conf in zip(window.boxes.xyxy, window.boxes.conf):
                x_min, y_min, x_max, y_max = box.tolist()
                xyxy = (x_min + window.x_offset, y_min, x_max + window.x_offset, y_max)
                boxes_xyxy_conf.append((xyxy, conf.item()))

        logger.info("working on board %s", self.id)

        boxes_xyxy = [xyxy for xyxy, conf in boxes_xyxy_conf]
        boxes_xywh = [
            xyxy_to_xywh(xyxy, self.image.shape[1], self.image.shape[0])
            for xyxy in boxes_xyxy
        ]

        return boxes_xyxy, boxes_xywh
