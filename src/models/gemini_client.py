import os
import google.generativeai as genai
from dotenv import load_dotenv
import logging
from src.utils.logger import setup_logger

load_dotenv()
logger = setup_logger("gemini_client")

class GeminiClient:
    """Wrapper for Google Gemini API to handle multimodal analysis."""
    
    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.warning("GOOGLE_API_KEY not found in .env. Multimodal vision tasks will fail.")
        else:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel("gemini-2.5-flash")
            logger.info("Gemini 2.5 Flash client initialized.")

    def summarize_image(self, image_data: bytes, prompt: str = "Describe this automotive diagram/image in detail for technical documentation.") -> str:
        """Sends an image to Gemini for technical description."""
        if not hasattr(self, 'model'):
            return "Vision analysis unavailable: No API key provided."
            
        try:
            response = self.model.generate_content([
                prompt,
                {"mime_type": "image/jpeg", "data": image_data}
            ])
            return response.text
        except Exception as e:
            logger.error(f"Gemini Vision error: {str(e)}", exc_info=True)
            return f"Error analyzing image: {str(e)}"
