import logging
import json
import os
from datetime import datetime
from typing import Any, Dict

class JsonFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields if provided
        if hasattr(record, "extra"):
            log_data.update(record.extra)
            
        return json.dumps(log_data)

def setup_logger(name: str, log_file: str = "logs/run_log.json") -> logging.Logger:
    """Sets up a logger that outputs to both console and a JSON file.
    
    Args:
        name: The name of the logger.
        log_file: The path to the JSON log file.
        
    Returns:
        A configured logging.Logger instance.
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # JSON File Handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(JsonFormatter())
    logger.addHandler(file_handler)
    
    # Console Handler (Standard Formatting)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)
    
    return logger
