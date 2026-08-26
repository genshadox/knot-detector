import logging
from pathlib import Path

from ultralytics import YOLO

logger = logging.getLogger(__name__)


class ModelRunner:
    def __init__(self, model_path: Path):

        self.model_path = model_path
        self.model = YOLO(str(self.model_path))

    def run_inference(self, input_data: list[Path], output_folder: Path) -> Path:

        for image_file in input_data:
            self.run_inference_single(image_file, output_folder)

        return output_folder

    def run_inference_single(self, image_file: Path, output_folder: Path) -> Path:

        try:
            self.model.predict(
                source=str(image_file),
                save=True,
                save_txt=True,
                project=str(output_folder),
                name=image_file.stem,
                exist_ok=True,
            )

        except Exception as e:
            logger.exception(f"An error occurred while processing {image_file}.")

        return output_folder
