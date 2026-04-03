import json
import os
import logging
from typing import List, Dict, Any, Optional
from llama_cpp import Llama
from src.utils.logger import setup_logger

logger = setup_logger("local_llm")

class LocalLLM:
    """Wrapper for llama-cpp-python to run 4-bit quantized models efficiently on CPU."""
    
    def __init__(self, config_path: str = "config/settings.json"):
        with open(config_path, "r") as f:
            self.config = json.load(f)["model"]
        
        self.model_path = self.config["local_path"]
        self.params = self.config["params"]
        self.llm = None
        
    def load_model(self):
        """Loads the model into memory with mmap for efficiency."""
        if not os.path.exists(self.model_path):
            logger.error(f"Model file not found at {self.model_path}. Please download it first.")
            raise FileNotFoundError(self.model_path)
            
        logger.info(f"Loading local LLM from {self.model_path}...")
        try:
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=self.params["n_ctx"],
                n_threads=self.params["n_threads"],
                use_mmap=self.params["mmap"],
                verbose=False
            )
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}", exc_info=True)
            raise

    def generate(self, prompt: str, system_prompt: str = "You are a professional automotive engineering assistant.") -> str:
        """Generates a response for a given prompt."""
        if self.llm is None:
            self.load_model()
            
        formatted_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        try:
            response = self.llm(
                formatted_prompt,
                max_tokens=self.params["max_tokens"],
                temperature=self.params["temperature"],
                top_p=self.params["top_p"],
                stop=["<|im_end|>"]
            )
            return response["choices"][0]["text"].strip()
        except Exception as e:
            logger.error(f"Generation error: {str(e)}", exc_info=True)
            return "I'm sorry, I encountered an error during response generation."

    def __del__(self):
        """Clean up resources on deletion."""
        if hasattr(self, 'llm') and self.llm:
            del self.llm
            import gc
            gc.collect()
