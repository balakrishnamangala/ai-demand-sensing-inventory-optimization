"""Centralized logging configuration using loguru."""
import sys
from pathlib import Path
from loguru import logger


def get_logger(name: str, level: str = "INFO"):
    """Configure and return a named logger instance.

    Args:
        name: Logger name for filtering and file routing.
        level: Minimum log level (DEBUG, INFO, WARNING, ERROR).

    Returns:
        Configured loguru logger bound to the given name.
    """
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
        level=level,
        filter=lambda record: record["extra"].get("name", "") == name,
    )
    logger.add(
        str(log_dir / f"{name}.log"),
        rotation="10 MB",
        retention="30 days",
        compression="gz",
        level=level,
        filter=lambda record: record["extra"].get("name", "") == name,
    )
    return logger.bind(name=name)
