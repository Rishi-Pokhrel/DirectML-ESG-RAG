# Project Implementation Plan: Automotive RAG System

## 1. Goal
Build a memory-efficient, multimodal RAG system for automotive repair and engineering (Transmissions, Drive Axles, General Maintenance) that can operate on ultra-low resource devices (250MB - 520MB RAM) using a 4-bit quantized 2B parameter LLM (~0.5GB).

## 2. Technical Architecture & Resource Optimization

### 2.1 Ultra-Low RAM Strategy (Target: 250MB - 520MB)
*   **LLM Engine:** Use `llama-cpp-python` with `mmap=True`. This allows the model to reside on disk and be mapped into memory only as needed, which is critical for a 500MB model on a 250MB RAM device.
*   **Vector DB:** Use **ChromaDB** with persistence on disk. We will query only the most relevant chunks to keep the context window small (e.g., 512 - 1024 tokens max).
*   **Embeddings:** Use an extremely lightweight local model (like `all-MiniLM-L6-v2` or similar) or Google Gemini API if internet access is permitted to offload CPU/RAM usage.
*   **Garbage Collection:** Explicitly call `gc.collect()` after each ingestion and query turn.
*   **Process Management:** Run FastAPI with a single worker to minimize memory footprint.

### 2.2 Multimodal Strategy
*   **Text:** Parse using `pypdf` (low memory overhead compared to `unstructured`).
*   **Tables:** Extract using `pdfplumber` or `tabula-py` only on demand.
*   **Images:** Use Gemini 1.5 Flash (Vision) via API to generate text-based summaries of diagrams (transmissions, axles) during ingestion. These summaries will be stored in the vector DB.

## 3. Data Processing (Automotive Domain)
*   **Sources:**
    1.  `AutoTrans_..._samplech11.pdf`: Automatic Transmission details.
    2.  `Crawfords_Auto_Repair_Guide.pdf`: General repair procedures.
    3.  `Passenger Car Drive Axle Technology.pdf`: Engineering specs for drive axles.
*   **Chunking:** Small, semantic chunks (300-500 tokens) with overlap to preserve technical context.

## 4. Required API Endpoints (FastAPI)
1.  `GET /health`: Returns system status and RAM usage.
2.  `POST /ingest`: Process one of the 3 PDFs and save summaries/embeddings to disk.
3.  `POST /query`: Perform RAG.
    *   Input: Natural language query (e.g., "How to troubleshoot a slipping transmission?").
    *   Output: Answer + Source Citation (Page #, File Name).
4.  `GET /documents`: List indexed automotive documents.

## 5. Implementation Checklist (Assignment.md Alignment)
- [ ] **Problem Statement:** Draft 500-800 words on "Automotive Diagnostic Support for Independent Mechanics."
- [ ] **Architecture Diagram:** Create Mermaid diagram in README.
- [ ] **Ingestion Module:** Implement memory-safe PDF parsing + Gemini Vision summaries.
- [ ] **Retrieval Module:** ChromaDB setup with persistent storage.
- [ ] **LLM Module:** `llama-cpp-python` wrapper for the 4-bit 2B model.
- [ ] **API Module:** FastAPI routes with Pydantic validation.
- [ ] **Screenshots:** Capture evidence of ingestion, text/table/image queries.

## 6. Recovery & Logging
- [ ] Integrate `src/utils/logger.py` into every module.
- [ ] Use `src/utils/state_manager.py` to track ingestion progress so we can resume if the codespace crashes during heavy parsing.
- [ ] Save all LLM parameters (top_p, temperature, etc.) in `config/settings.json`.
