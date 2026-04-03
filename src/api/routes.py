from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import psutil
import gc
from src.ingestion.processor import DocumentProcessor
from src.retrieval.engine import RAGEngine
from src.retrieval.vector_store import VectorStore

app = FastAPI(title="DirectML Automotive-RAG API")

# Mount static files
app.mount("/static", StaticFiles(directory="src/api/static"), name="static")

# Lazy load engines to save RAM at startup
processor = None
engine = None
vector_store = None

class QueryRequest(BaseModel):
    text: str

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[Dict[str, Any]]

@app.get("/")
def read_root():
    """Serves the web interface."""
    return FileResponse("src/api/static/index.html")

@app.get("/health")
def health():
    """Returns system status and memory metrics."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    
    global vector_store
    if vector_store is None:
        vector_store = VectorStore()
        
    return {
        "status": "online",
        "memory_usage_mb": mem_info.rss / (1024 * 1024),
        "indexed_documents": vector_store.get_count(),
        "available_ram_mb": psutil.virtual_memory().available / (1024 * 1024)
    }

@app.post("/ingest")
def ingest_files():
    """Ingests all PDFs from the sample_documents folder."""
    global processor
    if processor is None:
        processor = DocumentProcessor()
        
    source_dir = "sample_documents"
    files = [f for f in os.listdir(source_dir) if f.endswith(".pdf")]
    
    results = []
    for file_name in files:
        file_path = os.path.join(source_dir, file_name)
        processor.process_pdf(file_path)
        results.append(file_name)
    
    # Clean up processor immediately to free RAM for inference
    del processor
    processor = None
    gc.collect()
        
    return {"message": "Ingestion completed", "processed_files": results}

@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    """Answers an automotive engineering question with Adaptive Resource Management."""
    global engine, processor, vector_store
    
    # 1. Initialize Vector Store to check document count
    if vector_store is None:
        vector_store = VectorStore()
    
    # 2. Check RAM Specification (Adaptive Logic)
    # Threshold: 1024MB available RAM for safe "One-Click" Ingest + Inference
    available_ram_mb = psutil.virtual_memory().available / (1024 * 1024)
    
    if vector_store.get_count() == 0:
        if available_ram_mb < 1024:
            # SAFETY MODE: RAM too low for simultaneous ingest/inference request
            raise HTTPException(
                status_code=400, 
                detail="RESOURCE_LIMIT: Low RAM detected (Target 520MB hardware). Please use 'Re-Ingest' button first to stabilize system."
            )
        
        # ONE-CLICK MODE: High RAM detected (e.g. Codespace)
        if processor is None:
            processor = DocumentProcessor()
        
        source_dir = "sample_documents"
        if os.path.exists(source_dir):
            files = [f for f in os.listdir(source_dir) if f.endswith(".pdf")]
            for file_name in files:
                processor.process_pdf(os.path.join(source_dir, file_name))
            
            # Flush RAM immediately
            del processor
            processor = None
            gc.collect()
    
    # 3. Initialize Engine and Query
    if engine is None:
        engine = RAGEngine()
        
    try:
        response = engine.answer_query(request.text)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
