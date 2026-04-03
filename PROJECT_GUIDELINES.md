# Project Guidelines & Operational Mandates

This document serves as the primary instruction set for all development on the DirectML-ESG-RAG project. Every session must begin by acknowledging and adhering to these rules.

## 1. Hardware Optimization & Resource Management (8GB RAM / 12GB Disk)
- **Memory-Efficient Parsing:** Use generator-based processing for document ingestion. Never load entire large PDFs into memory.
- **Remote Processing:** Leverage Google Gemini API for heavy lifting (embeddings/LLM) to keep local CPU/RAM usage minimal.
- **Garbage Collection:** Explicitly trigger `gc.collect()` after memory-intensive operations (e.g., after processing a large PDF).
- **Disk Management:** Periodically clean up temporary files in `data/processed` and `data/raw`. Use ChromaDB's persistent storage efficiently.

## 2. Resilience & Error Handling
- **JSON Logging:** All runs must generate a `logs/run_log.json`. In case of a crash, the last state must be captured in this format.
- **Checkpointing:** Save ingestion progress (e.g., `last_processed_file`, `chunk_index`) in `data/state.json`.
- **Atomic Operations:** Ensure vector DB writes are atomic. If a write fails, the system must rollback or mark the document as "failed" to retry later.

## 3. Persistence & Recovery
- **Model Parameters:** Save all model configurations, prompt templates, and hyperparameters in `config/settings.json`.
- **Metrics Tracking:** Save performance metrics (latency, token counts, retrieval accuracy) alongside the results.
- **Resume Capability:** On startup, the system must check `data/state.json` and offer to resume from the last successful checkpoint.

## 4. Engineering Standards
- **Human-Centric Code:** Follow PEP 8 strictly. Use meaningful variable names, comprehensive docstrings (Google style), and type hinting.
- **Modular Architecture:** strictly follow the `src/` structure:
    - `src/ingestion/`: Logic for parsing and chunking.
    - `src/retrieval/`: Vector store interactions.
    - `src/models/`: Clean wrappers for Gemini/Vision models.
    - `src/api/`: Pydantic-validated FastAPI routes.
- **Validation:** Every feature must be verified against `data/assignment.md`.

## 5. Deployment & Structure
- **Paths:** Always use relative paths from the project root.
- **Secrets:** Use `.env` for all keys. Never hardcode.
- **Structure:** Maintain the layout specified in Section 4.1 of `assignment.md`.
