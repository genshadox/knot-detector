from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics.engine.results import Results

from util import save_labels, xyxy_to_xywh


def build_boards(paths: list[Path]) -> list["Board"]:
    """
    Build Board objects from a list of image paths.

    Args:
        paths (list[Path]): List of image file paths.

    Returns:
        list[Board]: List of Board objects with associated frames.
    """
    boards: dict[str, Board] = {}
    for path in paths:
        board_id, frame_num = path.stem.split("_")
        boards.setdefault(board_id, Board(board_id, [])).frames.append(
            Frame(path, int(frame_num))
        )

    result = list(boards.values())
    for board in result:
        board.frames.sort(key=lambda f: f.frame_num)

    return result


@dataclass
class Frame:
    """
    Represents a single frame in a board.
    """
    
    path: Path
    frame_num: int
    results: Results | None = None  # Results will be set after inference


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

    def frame_paths(self) -> list[Path]:
        """
        Get the paths of all frames in the board.

        Returns:
            list[Path]: List of paths for all frames in the board.
        """

        return [f.path for f in self.frames]

    def set_frame_results(self, frame_num: int, result: Results | None):
        """
        Set the results for a specific frame in the board.

        Args:
            frame_num (int): Frame number to set the results for.
            result (Results | None): Results object to associate with the frame.
        """

        for frame in self.frames:
            if frame.frame_num == frame_num:
                frame.results = result
                return

        raise ValueError(f"No frame {frame_num} in board {self.id}")

    def save_results(self, output_folder: Path):
        """
        Save the results of the board to the specified output folder.

        Args:
            output_folder (Path): Path to the folder where the results will be saved.
        """

        boxes_xyxy, boxes_xywh = self.to_board_space()

        save_labels(boxes_xywh, self.CLS, output_folder / f"{self.id}.txt")

        self.plot_board(
            self.frame_paths(), boxes_xyxy, output_folder / f"{self.id}.png"
        )

    def plot_board(
        self, image_paths: list[Path], boxes_xyxy: list[tuple], output_path: Path
    ):
        """
        Plot the board by stitching frames together and drawing bounding boxes.

        Args:
            image_paths (list[Path]): List of paths to the frame images.
            boxes_xyxy (list[tuple]): List of bounding boxes in xyxy format.
            output_path (Path): Path to save the plotted board image.
        """

        images = [cv2.imread(str(p)) for p in image_paths]
        stitched = np.hstack(images)  # same height assumed across frames

        for x_min, y_min, x_max, y_max in boxes_xyxy:
            cv2.rectangle(
                stitched,
                (int(x_min), int(y_min)),
                (int(x_max), int(y_max)),
                color=(0, 255, 0),
                thickness=2,
            )

        cv2.imwrite(str(output_path), stitched)

    def to_board_space(self) -> tuple[list, list]:
        """
        Convert bounding boxes from individual frame coordinates to board coordinates.

        Returns:
            tuple[list, list]: A tuple containing two lists:
                - List of bounding boxes in xyxy format.
                - List of bounding boxes in xywh format, normalized by board width and height.
        """


        width = height = 0
        boxes_xyxy = []

        for frame in self.frames:

            assert frame.results is not None

            result = frame.results
            height = max(height, result.orig_shape[0])

            for box in result.boxes.xyxy:
                x_min, y_min, x_max, y_max = box.tolist()
                boxes_xyxy.append((x_min + width, y_min, x_max + width, y_max))

            width += result.orig_shape[1]

        boxes_xywh = [xyxy_to_xywh(xyxy, width, height) for xyxy in boxes_xyxy]
        return boxes_xyxy, boxes_xywh
