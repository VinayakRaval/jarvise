import logging
import sys
from typing import Dict


_loggers: Dict[str, logging.Logger] = {}


def get_logger(name: str = "JARVIS") -> logging.Logger:
    """
    Return a configured logger.

    The same logger instance is reused when requested
    multiple times with the same name.
    """

    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)

    # Prevent duplicate handlers
    logger.propagate = False

    if not logger.handlers:

        handler = logging.StreamHandler(sys.stdout)

        formatter = logging.Formatter(
            fmt=(
                "%(asctime)s | "
                "%(levelname)s | "
                "%(name)s | "
                "%(message)s"
            ),
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        handler.setFormatter(formatter)

        logger.addHandler(handler)

    logger.setLevel(logging.INFO)

    _loggers[name] = logger

    return logger