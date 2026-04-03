"""
Utilities for AutoDiag.
Includes JSON logging and state management for crash recovery.
"""

from .logger import setup_logger
from .state_manager import StateManager

__all__ = ["setup_logger", "StateManager"]
