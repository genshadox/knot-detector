import logging
import os
import sys
from pathlib import Path

from logger import setup_logging
from model_runner import ModelRunner
from test import test_results
from util import (
    get_input_path,
    get_model_path,
    get_output_path,
    get_test_input_path,
)

logger = logging.getLogger(__name__)


def run_inference(model_runner: ModelRunner, input_path: Path, output_path: Path):
    """
    Run inference on a list of input images and save the results per board to the specified output folder.

    Args:
        model_runner (ModelRunner): An instance of the ModelRunner class.
        input_path (Path): Path to the folder containing input images.
        output_path (Path): Path to the folder where inference results will be saved.
    """

    image_files = list(input_path.glob("*.png"))

    model_runner.run_inference(input_data=image_files, output_folder=output_path)


def main():
    """
    Main function to run the inference process based on the specified mode (test or normal).
    """

    mode = os.environ["MODE"]

    model_runner = ModelRunner(model_path=get_model_path())

    if mode == "test":
        logger.info(
            "Running in test mode. Inference results will be saved to the 'compare' folder."
        )

        test_input_path = get_test_input_path()
        test_input_path_compare = test_input_path / "compare"

        run_inference(
            model_runner=model_runner,
            input_path=test_input_path,
            output_path=test_input_path_compare,
        )

        test_results(test_input_path, test_input_path_compare)

    else:
        logger.info(
            "Running in normal mode. Inference results will be saved to the 'output' folder."
        )

        run_inference(
            model_runner=model_runner,
            input_path=get_input_path(),
            output_path=get_output_path(),
        )

    logger.info("done")


if __name__ == "__main__":
    try:
        setup_logging()
        main()

    except Exception:
        logger.exception("Script failed with an unhandled exception.")
        sys.exit(1)
