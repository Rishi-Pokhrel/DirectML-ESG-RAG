"""
Models module for AutoDiag.
Wrappers for Local LLM (llama-cpp) and Gemini Vision API.
"""

from .local_llm import LocalLLM
from .gemini_client import GeminiClient

__all__ = ["LocalLLM", "GeminiClient"]
