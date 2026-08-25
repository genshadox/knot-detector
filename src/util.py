from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).parent


def get_data_dir() -> Path:
    return Path("/data")


def get_logs_path() -> Path:
    return get_data_dir() / "logs"


def get_model_path() -> Path:
    return get_project_root() / "model" / "weights" / "best.pt"


def get_output_path() -> Path:
    return get_data_dir() / "output"


def get_input_path() -> Path:
    return get_data_dir() / "input"
