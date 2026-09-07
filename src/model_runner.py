import logging
from pathlib import Path

import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

from board import build_boards

logger = logging.getLogger(__name__)


class ModelRunner:
    WINDOW_SIZE = 640
    WINDOW_STRIDE = 320

    def __init__(self, model_path: Path):
        """
        Initialize the ModelRunner with the specified model path.

        Args:
            model_path (Path): Path to the YOLO model weights file.
        """

        self.model_path: Path = model_path
        self.model: YOLO = YOLO(str(self.model_path))

    def run_inference(self, input_data: list[Path], output_folder: Path) -> Path:
        """
        Run inference on a list of input images and save the results per board to the specified output folder.

        Args:
            input_data (list[Path]): List of image file paths to run inference on.
            output_folder (Path): Path to the folder where inference results will be saved.

        Returns:
            Path: Path to the folder where inference results are saved.
        """

        boards = build_boards(input_data)

        for board in boards:
            try:
                x = 0

                while x < board.image.shape[1]:
                    window_data = board.get_window_data(x, self.WINDOW_SIZE)

                    results = self.run_inference_single(
                        window_data, board.id, output_folder
                    )

                    if results.boxes is None:
                        logger.warning(
                            f"No boxes attribute on window result for board {board.id}, x={x} — skipping window."
                        )
                        x += self.WINDOW_STRIDE
                        continue

                    board.add_window(x, results.orig_shape[1], boxes=results.boxes)

                    x += self.WINDOW_STRIDE

                board.save_results(output_folder=output_folder)

            except Exception:
                logger.exception(
                    f"An error occurred while processing board {board.id}."
                )

        return output_folder

    def run_inference_single(
        self, image: np.ndarray, name: str, output_folder: Path
    ) -> Results:
        """
        Run inference on a single image and save the results to the specified output folder.

        Args:
            image (np.ndarray): The image data as a NumPy array.
            name (str): Name to use for saving the results.
            output_folder (Path): Path to the folder where inference results will be saved.

        Returns:
            Results: The inference results for the image.
        """

        try:
            results = self.model.predict(
                source=image,
                # save=True,
                # save_txt=True,
                project=str(output_folder),
                name=name,
                exist_ok=True,
                iou=0.5,
            )

            return results[0]  # Only 1 frame so only 1 result (with multiple boxes)

        except Exception:
            logger.exception("An error occurred while processing.")
            raise
