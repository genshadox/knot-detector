import logging
import sys
from logging.handlers import RotatingFileHandler

from util import get_logs_path


def setup_logging():
    """
    Set up logging configuration.
    """

    log_path = get_logs_path()
    log_path.mkdir(parents=True, exist_ok=True)

    # Configure rotating file handler for our main log file
    rotating_handler = RotatingFileHandler(
        log_path / 'application.log',
        maxBytes=20*1024*1024,      # 20 MB
        backupCount=5,              # Keep 5 backups
        encoding='utf8'
    )

    # Create a formatter that includes timestamp and log level
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    rotating_handler.setFormatter(formatter)

    # Get root logger and configure it
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture all levels of logs

    # Clear any existing handlers to avoid duplicates
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Add rotating file handler
    root_logger.addHandler(rotating_handler)

    # Also add console handler so logs appear in Docker output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)