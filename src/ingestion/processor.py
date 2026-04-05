import os
import gc
import logging
import re
from typing import List, Dict, Any
import pypdf
import pdfplumber
from PIL import Image
import io
from src.utils.logger import setup_logger
from src.utils.state_manager import StateManager
from src.retrieval.vector_store import VectorStore
from src.models.gemini_client import GeminiClient

logger = setup_logger("ingestion")

class DocumentProcessor:
    """Processes multimodal PDFs with Precision Micro-Chunking for high-density RAG."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100, max_images_per_pdf: int = 5):
        self.vector_store = VectorStore()
        self.state_manager = StateManager()
        self.gemini = GeminiClient()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_images_per_pdf = max_images_per_pdf
        self.images_processed_count = 0

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

    def _extract_tables_from_page(self, pdf_path: str, page_num: int) -> List[str]:
        """Extract tables from a specific page using pdfplumber."""
        formatted_tables = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                if page_num < len(pdf.pages):
                    page = pdf.pages[page_num]
                    tables = page.extract_tables()
                    
                    for table_idx, table in enumerate(tables):
                        if table:
                            # Convert table to readable text format
                            table_text = f"TABLE {table_idx + 1}:\n"
                            for row in table:
                                # Clean and join cells
                                cleaned_row = [str(cell).strip() if cell else "" for cell in row]
                                table_text += " | ".join(cleaned_row) + "\n"
                            formatted_tables.append(table_text.strip())
                            logger.info(f"Extracted table {table_idx + 1} from page {page_num + 1}")
        except Exception as e:
            logger.warning(f"Error extracting tables from page {page_num + 1}: {str(e)}")
        
        return formatted_tables

    def _extract_and_analyze_images(self, file_path: str, page_num: int, file_name: str) -> List[Dict]:
        """Extract images from PDF page and analyze with Gemini Vision (with rate limiting)."""
        images_analyzed = []
        
        # Skip if we've already processed max images for this PDF
        if self.images_processed_count >= self.max_images_per_pdf:
            return images_analyzed
        
        try:
            reader = pypdf.PdfReader(file_path)
            if page_num >= len(reader.pages):
                return images_analyzed
                
            page = reader.pages[page_num]
            
            # Check if page has images
            if "/Resources" not in page or "/XObject" not in page["/Resources"]:
                return images_analyzed
            
            xObject = page["/Resources"]["/XObject"].get_object()
            
            for obj_idx, obj_name in enumerate(xObject):
                try:
                    obj = xObject[obj_name]
                    
                    # Check if it's an image
                    if obj.get("/Subtype") == "/Image":
                        # Stop if we've hit our limit
                        if self.images_processed_count >= self.max_images_per_pdf:
                            logger.info(f"Reached max images limit ({self.max_images_per_pdf}) for this PDF")
                            break
                        
                        # Extract image data
                        data = obj.get_data()
                        
                        # Only analyze if we have Gemini API configured
                        if hasattr(self.gemini, 'model'):
                            try:
                                description = self.gemini.summarize_image(
                                    data,
                                    prompt="Describe this automotive technical diagram or image in detail. Include: components shown, relationships, measurements, and technical specifications visible."
                                )
                                logger.info(f"Analyzed image {obj_idx + 1} on page {page_num + 1} with Gemini Vision")
                                self.images_processed_count += 1
                            except Exception as e:
                                logger.warning(f"Gemini Vision failed for image on page {page_num + 1}: {str(e)}")
                                description = f"Technical diagram from automotive manual (page {page_num + 1})"
                        else:
                            # Fallback if no API key
                            description = f"Technical diagram or image from automotive manual on page {page_num + 1}. Visual content detected but not analyzed (requires GOOGLE_API_KEY)."
                        
                        images_analyzed.append({
                            "description": description,
                            "metadata": {
                                "source": file_name,
                                "page": page_num + 1,
                                "type": "image",
                                "image_index": obj_idx
                            }
                        })
                        
                except Exception as e:
                    logger.warning(f"Could not extract image object on page {page_num + 1}: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error processing images from page {page_num + 1}: {str(e)}")
        
        return images_analyzed

    def process_pdf(self, file_path: str):
        """Extracts content from PDF using micro-chunking for higher retrieval precision."""
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return
            
        # Reset image counter for each PDF
        self.images_processed_count = 0
            
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
            # 1. Extract and chunk text
            text = page.extract_text()
            if text.strip():
                # Perform micro-chunking per page
                micro_chunks = self._get_micro_chunks(text)
                for j, m_chunk in enumerate(micro_chunks):
                    chunk_id = f"{file_name}_p{i}_text_c{j}"
                    all_chunks.append(m_chunk)
                    all_metadatas.append({
                        "source": file_name, 
                        "page": i + 1, 
                        "type": "text",
                        "chunk_index": j
                    })
                    all_ids.append(chunk_id)

            # 2. Extract tables from page
            tables = self._extract_tables_from_page(file_path, i)
            for table_idx, table_text in enumerate(tables):
                chunk_id = f"{file_name}_p{i}_table_{table_idx}"
                all_chunks.append(table_text)
                all_metadatas.append({
                    "source": file_name,
                    "page": i + 1,
                    "type": "table",
                    "table_index": table_idx
                })
                all_ids.append(chunk_id)

            # 3. Extract and analyze images with Gemini Vision
            image_results = self._extract_and_analyze_images(file_path, i, file_name)
            for img_data in image_results:
                chunk_id = f"{file_name}_p{i}_image_{img_data['metadata']['image_index']}"
                all_chunks.append(img_data['description'])
                all_metadatas.append(img_data['metadata'])
                all_ids.append(chunk_id)

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
