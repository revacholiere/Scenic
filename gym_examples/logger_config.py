import logging

import colorlog

from praeception.config.config import cfg


SUPPRESS = logging.CRITICAL + 1


class CustomLogRecord(logging.LogRecord):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.service = cfg.service
        self.process_index = cfg.process_index


logging.setLogRecordFactory(CustomLogRecord)


def configure_logger(log_level):
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Console handler
    ch = colorlog.StreamHandler()
    ch.setLevel(log_level)

    # Colors
    service_color = "\033[35m"
    reset_color = "\033[0m"
    colors = {
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "red,bg_white",
    }

    # Formatter
    formatter = colorlog.ColoredFormatter(
        f"{service_color}[%(service)s:%(process_index)02d]{reset_color} %(log_color)s%(asctime)s %(levelname)-10s%(reset)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        reset=True,
        log_colors=colors,
        style="%",
    )
    ch.setFormatter(formatter)

    # Add handler
    logger.addHandler(ch)

    return logger
