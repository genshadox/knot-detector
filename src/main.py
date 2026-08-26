import logging
import sys

from logger import setup_logging
from model_runner import ModelRunner
from util import get_input_path, get_model_path, get_output_path

logger = logging.getLogger(__name__)


def main():

    input_path = get_input_path()
    output_path = get_output_path()

    image_files = list(input_path.glob("*.png"))

    model_runner = ModelRunner(model_path=get_model_path())
    model_runner.run_inference(input_data=image_files, output_folder=output_path)


    logger.info("done")


if __name__ == "__main__":
    try:
        setup_logging()
        main()

    except Exception:
        logger.exception("Script failed with an unhandled exception.")
        sys.exit(1)
