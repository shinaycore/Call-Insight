import logging
import os
import inspect

class ColorFormatter(logging.Formatter):
    COLOR_MAP = {
        'DEBUG': "\033[36m",    # Cyan
        'INFO': "\033[34m",     # Blue
        'WARNING': "\033[33m",  # Yellow
        'ERROR': "\033[31m",    # Red
        'CRITICAL': "\033[41m", # Red background
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLOR_MAP.get(record.levelname, self.RESET)
        message = super().format(record)
        return f"{color}{message}{self.RESET}"

def get_logger(name=None, level=logging.INFO) -> logging.Logger:
    if name is None:
        # Automatically get the caller file name without extension
        frame = inspect.stack()[1]
        name = os.path.splitext(os.path.basename(frame.filename))[0]

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = ColorFormatter("[%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger
