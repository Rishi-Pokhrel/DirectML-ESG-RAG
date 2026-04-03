from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import psutil
from src.ingestion.processor import DocumentProcessor
from src.retrieval.engine import RAGEngine
from src.retrieval.vector_store import VectorStore

app = FastAPI(title="Automotive Multimodal RAG API")

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
    """Ingests all PDFs from the data/raw folder."""
    global processor
    if processor is None:
        processor = DocumentProcessor()
        
    raw_dir = "data/raw"
    files = [f for f in os.listdir(raw_dir) if f.endswith(".pdf")]
    
    results = []
    for file_name in files:
        file_path = os.path.join(raw_dir, file_name)
        processor.process_pdf(file_path)
        results.append(file_name)
        
    return {"message": "Ingestion completed", "processed_files": results}

@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    """Answers an automotive engineering question. Auto-ingests if DB is empty."""
    global engine, processor, vector_store
    
    # 1. Initialize Vector Store to check count
    if vector_store is None:
        vector_store = VectorStore()
    
    # 2. Auto-Ingest if no documents exist
    if vector_store.get_count() == 0:
        if processor is None:
            processor = DocumentProcessor()
        
        # Use sample_documents as the authoritative source for auto-ingest
        source_dir = "sample_documents"
        if os.path.exists(source_dir):
            files = [f for f in os.listdir(source_dir) if f.endswith(".pdf")]
            for file_name in files:
                processor.process_pdf(os.path.join(source_dir, file_name))
    
    # 3. Initialize Engine and Query
    if engine is None:
        engine = RAGEngine()
        
    try:
        response = engine.answer_query(request.text)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
