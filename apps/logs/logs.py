from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional
import logging


def get_log_base_dir() -> Path:
    """Resolve the log directory relative to this file (apps/logs)."""
    log_dir = Path(__file__).resolve().parent
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


LOG_BASE_DIR = get_log_base_dir()
LOG_FILE = LOG_BASE_DIR / "app.log"

_file_handler: Optional[RotatingFileHandler] = None


def get_file_handler(name: str = "app", max_bytes: int = 5 * 1024 * 1024, backup_count: int = 1) -> RotatingFileHandler:
    """
    Return a shared file handler so all modules log into the same file while the
    logger name keeps entries identifiable.
    """
    global _file_handler
    _ = name  # kept for compatibility; logger name in formatter distinguishes entries
    if _file_handler is None:
        handler = RotatingFileHandler(
            filename=LOG_FILE,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8"
        )
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        handler.setLevel(logging.DEBUG)
        _file_handler = handler
    return _file_handler


console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
))


def get_logger(name: str) -> logging.Logger:
    """
    Fetch a logger that writes to the shared file and console.
    The logger name distinguishes entries across modules.
    """
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)
        logger.addHandler(get_file_handler(name))
        logger.addHandler(console_handler)
        logger.propagate = False
    return logger


if __name__ == "__main__":
    demo_logger = get_logger(__name__)
    demo_logger.info("Logger initialized")
