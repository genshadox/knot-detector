import logging
import sys

from logger import setup_logging

logger = logging.getLogger(__name__)

def main():
    logger.info('done')

if __name__ == "__main__":
    try:
        setup_logging()
        main()

    except Exception:
        logger.exception("Script failed with an unhandled exception.")
        sys.exit(1)
