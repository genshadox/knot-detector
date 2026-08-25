from pathlib import Path


def get_data_dir() -> Path:
    return Path('/data')


def get_logs_path() -> Path:
    return get_data_dir() / 'logs'

