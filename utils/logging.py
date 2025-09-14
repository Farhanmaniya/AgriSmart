"""
Logging configuration for AgriSmart backend.
Sets up structured logging for monitoring and debugging.
"""

import logging
import sys
import os
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str,
    level: Optional[str] = None,
    log_file: Optional[str] = None
) -> logging.Logger:
    """Setup logger with consistent formatting."""
    
    # Get log level from environment or default to INFO
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO").upper()
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        # Create logs directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def log_request(logger: logging.Logger, method: str, endpoint: str, user_id: Optional[str] = None):
    """Log API request."""
    user_info = f"User: {user_id}" if user_id else "Anonymous"
    logger.info(f"{method} {endpoint} - {user_info}")


def log_error(logger: logging.Logger, error: Exception, context: Optional[str] = None):
    """Log error with context."""
    error_msg = f"Error: {str(error)}"
    if context:
        error_msg = f"{context} - {error_msg}"
    logger.error(error_msg, exc_info=True)


def log_ml_prediction(logger: logging.Logger, prediction_type: str, user_id: str, confidence: float):
    """Log ML prediction."""
    logger.info(f"ML Prediction - Type: {prediction_type}, User: {user_id}, Confidence: {confidence:.2f}")


def log_irrigation_schedule(logger: logging.Logger, user_id: str, duration: int, water_volume: float):
    """Log irrigation scheduling."""
    logger.info(f"Irrigation Scheduled - User: {user_id}, Duration: {duration}min, Volume: {water_volume}L")