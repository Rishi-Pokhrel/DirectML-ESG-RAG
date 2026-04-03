import json
import os
from typing import Any, Dict, Optional
import logging

class StateManager:
    """Manages system state and checkpointing to ensure recovery after crashes."""
    
    def __init__(self, state_file: str = "data/state.json"):
        self.state_file = state_file
        self.logger = logging.getLogger(__name__)
        self._ensure_state_dir()

    def _ensure_state_dir(self):
        """Ensures the directory for the state file exists."""
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)

    def save_state(self, state: Dict[str, Any]):
        """Saves the current state to the state file.
        
        Args:
            state: A dictionary representing the system state.
        """
        try:
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=4)
            self.logger.info("State saved successfully", extra={"extra": {"state_file": self.state_file}})
        except Exception as e:
            self.logger.error(f"Failed to save state: {str(e)}")

    def load_state(self) -> Dict[str, Any]:
        """Loads the saved state from the state file.
        
        Returns:
            A dictionary of the saved state, or an empty dict if not found.
        """
        if not os.path.exists(self.state_file):
            return {}
        
        try:
            with open(self.state_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load state: {str(e)}")
            return {}

    def update_checkpoint(self, key: str, value: Any):
        """Updates a specific key in the state and saves immediately.
        
        Args:
            key: The key to update.
            value: The value to set.
        """
        state = self.load_state()
        state[key] = value
        self.save_state(state)
