import logging
import logging.config
from pathlib import Path

from .utils import read_json, ROOT_PATH


def setup_logging(
        save_dir, log_config=None, default_level=logging.INFO
):
    """
    Setup logging configuration
    """
    if log_config is None:
        log_config = str(ROOT_PATH / "multitask_learning_ASR_KWS" / "logger" / "logger_config.json")
    log_config = Path(log_config)
    if log_config.is_file():
        config = read_json(log_config)
        # modify logging paths based on run config
        for _, handler in config["handlers"].items():
            if "filename" in handler:
                handler["filename"] = str(save_dir / handler["filename"])

        logging.config.dictConfig(config)
    else:
        print(
            "Warning: logging configuration file is not found in {}.".format(log_config)
        )
        logging.basicConfig(level=default_level)
