from logging.handlers import RotatingFileHandler
from pathlib import Path
import logging
import sys


# ---------------- 日志基础目录 ----------------
def get_log_base_dir() -> Path:
    """
    返回日志目录路径：
    - 如果是 PyInstaller 打包后的 exe，使用 exe 所在目录/logs
    - 如果是源码运行，使用项目根目录/logs（即当前文件上两级）
    """
    if getattr(sys, 'frozen', False):
        # PyInstaller 打包环境，sys.executable 是 exe 文件路径
        base_dir = Path(sys.executable).parent
    else:
        # 普通 Python 环境，回到上两级目录
        base_dir = Path(__file__).resolve().parents[2]

    log_dir = base_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


LOG_BASE_DIR = get_log_base_dir()


# ---------------- 文件 Handler 创建函数 ----------------
def get_file_handler(name: str, max_bytes: int = 5 * 1024 * 1024, backup_count: int = 1) -> RotatingFileHandler:
    """
    根据 API 或模块名动态创建 RotatingFileHandler
    日志文件自动保存到不同文件夹
    """
    log_file = LOG_BASE_DIR / f"{name}.log"

    handler = RotatingFileHandler(
        filename=log_file,
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
    return handler


# ---------------- 控制台 Handler（共享） ----------------
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
))


# ---------------- 获取 logger 函数 ----------------
def get_logger(name: str) -> logging.Logger:
    """
    获取 logger：
    - 自动创建 Handler（按模块/API 名）
    - 添加到 logger（只添加一次）
    - 支持控制台输出
    """
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)
        logger.addHandler(get_file_handler(name))
        logger.addHandler(console_handler)
    return logger


if __name__ == '__main__':
    logger = get_logger("main")
