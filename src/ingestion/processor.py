import os
import gc
import logging
import re
from typing import List, Dict, Any
import pypdf
from src.utils.logger import setup_logger
from src.utils.state_manager import StateManager
from src.retrieval.vector_store import VectorStore
from src.models.gemini_client import GeminiClient

logger = setup_logger("ingestion")

class DocumentProcessor:
    """Processes multimodal PDFs with Precision Micro-Chunking for high-density RAG."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.vector_store = VectorStore()
        self.state_manager = StateManager()
        self.gemini = GeminiClient()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _get_micro_chunks(self, text: str) -> List[str]:
        """Splits text into small, overlapping segments to maintain context density."""
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start += (self.chunk_size - self.chunk_overlap)
        return chunks

    def process_pdf(self, file_path: str):
        """Extracts content from PDF using micro-chunking for higher retrieval precision."""
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return
            
        file_name = os.path.basename(file_path)
        logger.info(f"Starting Precision Ingestion of {file_name}")
        
        state = self.state_manager.load_state()
        if file_name in state.get("processed_files", []):
            logger.info(f"Skipping {file_name}, already processed.")
            return

        reader = pypdf.PdfReader(file_path)
        all_chunks = []
        all_metadatas = []
        all_ids = []
        
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text.strip():
                # Perform micro-chunking per page
                micro_chunks = self._get_micro_chunks(text)
                for j, m_chunk in enumerate(micro_chunks):
                    chunk_id = f"{file_name}_p{i}_c{j}"
                    all_chunks.append(m_chunk)
                    all_metadatas.append({
                        "source": file_name, 
                        "page": i + 1, 
                        "type": "text",
                        "chunk_index": j
                    })
                    all_ids.append(chunk_id)

            if "/XObject" in page["/Resources"]:
                logger.info(f"Detected image on page {i+1} of {file_name}")

            # Batch add to vector store every 5 pages to keep memory low
            if i % 5 == 0 and all_chunks:
                self.vector_store.add_documents(all_chunks, all_metadatas, all_ids)
                all_chunks, all_metadatas, all_ids = [], [], []
                gc.collect()

        # Add remaining chunks
        if all_chunks:
            self.vector_store.add_documents(all_chunks, all_metadatas, all_ids)
            
        # Update State
        processed_files = state.get("processed_files", [])
        processed_files.append(file_name)
        self.state_manager.update_checkpoint("processed_files", processed_files)
        logger.info(f"Finished Precision Ingestion of {file_name}")
        
        gc.collect()
