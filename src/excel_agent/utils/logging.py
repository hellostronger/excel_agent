"""Logging configuration for the Excel Intelligent Agent System."""

import logging
import sys
from typing import Optional
from loguru import logger
from .config import get_config


class InterceptHandler(logging.Handler):
    """Intercepts standard logging records and redirects to loguru."""
    
    def emit(self, record):
        # Get corresponding loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging():
    """Setup logging configuration."""
    # Remove default loguru handler
    logger.remove()
    
    # Add custom handler with format
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
        level=get_config().log_level,
        colorize=True,
    )
    
    # Add file handler for persistent logging
    logger.add(
        f"{get_config().temp_dir}/excel_agent.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        level="DEBUG",
        rotation="10 MB",
        retention="10 days",
    )
    
    # Intercept standard logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    
    # Suppress some noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def get_logger(name: Optional[str] = None):
    """Get a logger instance."""
    if name:
        return logger.bind(name=name)
    return logger


# Setup logging on import
setup_logging()