import logging
import sys
from typing import Optional

# ANSI Escape Codes for Colors
class LogColors:
    GREY = "\x1b[38;20m"
    BLUE = "\x1b[34;20m"
    CYAN = "\x1b[36;20m"
    YELLOW = "\x1b[33;20m"
    RED = "\x1b[31;20m"
    BOLD_RED = "\x1b[31;1m"
    RESET = "\x1b[0m"
    BOLD = "\x1b[1m"

class ColoredFormatter(logging.Formatter):
    """Custom Formatter to add colors to logs based on level."""
    
    # Format: [TIME] [LEVEL] [MODULE] - MESSAGE
    FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
    
    LEVEL_COLORS = {
        logging.DEBUG: LogColors.GREY,
        logging.INFO: LogColors.CYAN,
        logging.WARNING: LogColors.YELLOW,
        logging.ERROR: LogColors.RED,
        logging.CRITICAL: LogColors.BOLD_RED,
    }

    # Logger name aliasing for cleaner output
    LOGGER_ALIASES = {
        "uvicorn.error": "uvicorn.server",
        "uvicorn.access": "uvicorn.http",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(datefmt="%H:%M:%S", *args, **kwargs)

    def format(self, record):
        log_color = self.LEVEL_COLORS.get(record.levelno, LogColors.RESET)
        
        # Colorize the level name
        levelname = f"{log_color}{record.levelname:8}{LogColors.RESET}"
        
        # Colorize the time
        asctime = f"{LogColors.GREY}[{self.formatTime(record, self.datefmt)}]{LogColors.RESET}"
        
        # Alias and colorize internal module path
        log_name = self.LOGGER_ALIASES.get(record.name, record.name)
        name = f"{LogColors.BLUE}{log_name}{LogColors.RESET}"
        
        # Handle funcName/lineno with color
        func = f"{LogColors.GREY}{record.funcName}:{record.lineno}{LogColors.RESET}"

        # Combine into custom format
        message = record.getMessage()
        if record.levelno >= logging.WARNING:
            message = f"{log_color}{message}{LogColors.RESET}"

        return f"{asctime} | {levelname} | {name}:{func} - {message}"

def setup_logging(level: int = logging.INFO):
    """Initializes the root logger with colored output."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColoredFormatter())
    
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates during reload
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
        
    root_logger.addHandler(handler)
    
    # Reduce noise from third-party libraries
    logging.getLogger("uvicorn").handlers.clear()
    logging.getLogger("uvicorn.access").handlers.clear()
    
    # Prevent uvicorn from overriding our beautiful logs
    uvicorn_logger = logging.getLogger("uvicorn")
    uvicorn_logger.propagate = True
    uvicorn_error_logger = logging.getLogger("uvicorn.error")
    uvicorn_error_logger.propagate = True
    uvicorn_access_logger = logging.getLogger("uvicorn.access")
    uvicorn_access_logger.propagate = False # we might want to manually handle access or just let it be

def get_logger(name: str) -> logging.Logger:
    """Helper to get a logger with a specific name."""
    return logging.getLogger(name)
