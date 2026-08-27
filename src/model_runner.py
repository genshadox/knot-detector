import logging
from pathlib import Path

from ultralytics import YOLO
from ultralytics.engine.results import Results

from board import build_boards

logger = logging.getLogger(__name__)


class ModelRunner:
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
                for frame in board.frames:
                    results = self.run_inference_single(
                        frame.path, board.id, output_folder
                    )

                    board.set_frame_results(frame.frame_num, results)

                board.save_results(output_folder=output_folder)

            except Exception:
                logger.exception(
                    f"An error occurred while processing board {board.id}."
                )

        return output_folder

    def run_inference_single(
        self, image_file: Path, name: str, output_folder: Path
    ) -> Results:
        """
        Run inference on a single image and save the results to the specified output folder.

        Args:
            image_file (Path): Path to the image file to run inference on.
            name (str): Name to use for saving the results.
            output_folder (Path): Path to the folder where inference results will be saved.

        Returns:
            Results: The inference results for the image.
        """

        try:
            results = self.model.predict(
                source=str(image_file),
                save=True,
                save_txt=True,
                project=str(output_folder),
                name=name,
                exist_ok=True,
                iou=0.5,
            )

            return results[0]  # Only 1 frame so only 1 result (with multiple boxes)

        except Exception:
            logger.exception(f"An error occurred while processing {image_file}.")
            raise
