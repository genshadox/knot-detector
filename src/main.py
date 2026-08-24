import logging
import sys

logger = logging.getLogger(__name__)

def main():
    print('done')

if __name__ == "__main__":
    try:
        main()

    except Exception:
        logger.exception("Script failed with an unhandled exception.")
        sys.exit(1)
