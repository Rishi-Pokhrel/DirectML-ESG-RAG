# Multimodal RAG System with FastAPI - Assignment Requirements

## 1. Objective
Design, build, and deploy an end-to-end Multimodal RAG system that ingests multimodal PDFs (text, tables, images), builds a searchable vector index, and exposes a query interface via FastAPI.

## 2. Problem Statement (README.md)
*   **Location:** `README.md` under "Problem Statement" section.
*   **Length:** 500 - 800 words.
*   **Contents:**
    *   Domain Identification (e.g., automotive, healthcare, legal).
    *   Problem Description (current struggles with multimodal documents).
    *   Uniqueness (domain-specific challenges like specialized terminology, diagrams).
    *   RAG Justification (why RAG over fine-tuning or manual search).
    *   Expected Outcomes (queries it should answer).

## 3. Technical Requirements

### 3.1 Core System Capabilities
*   Ingest multimodal PDFs (Text, Tables, Images).
*   Build a searchable vector index.
*   FastAPI server for querying.

### 3.2 Required API Endpoints
*   `POST /ingest`: Upload and index a multimodal PDF.
*   `POST /query`: Query the index (must handle text, table, and image-based queries).
*   `GET /health`: System status and indexed document count.
*   *(Optional but recommended)*: `/documents` (list files), `/delete` (remove document).

### 3.3 Sample Domain PDF
*   Must include at least one PDF in `sample_documents/` with:
    *   Text content.
    *   At least one table.
    *   At least one image, chart, or diagram.

### 3.4 Technology Stack (Flexible)
*   **Parser:** Docling, Unstructured, PyMuPDF, etc.
*   **Embeddings:** Google Gemini, OpenAI, HuggingFace, etc.
*   **Vector Store:** ChromaDB, FAISS, Pinecone, etc.
*   **LLM/VLM:** Gemini 1.5 Pro/Flash, GPT-4o, Llama, etc.

## 4. Repository Structure
```text
your-repo-name/
├── README.md                  # Problem statement, arch diagram, tech choices, setup
├── requirements.txt           # Pinned Python dependencies
├── .env.example               # Template for API keys
├── main.py                    # FastAPI application entry point
├── src/                       # Source code (modular)
│   ├── ingestion/             # Parsing, chunking, embedding
│   ├── retrieval/             # Vector store, retriever, query logic
│   ├── models/                # LLM and Vision model wrappers
│   └── api/                   # FastAPI route definitions
├── sample_documents/          # Multimodal PDF(s)
├── screenshots/               # Evidence of working system
└── .gitignore                 # Excludes .env, __pycache__, etc.
```

## 5. Deliverables & Documentation
*   **Architecture Diagram:** In `README.md` (Mermaid or image).
*   **Technology Choices:** Justification for each component.
*   **Setup Instructions:** Step-by-step guide.
*   **API Documentation:** Description of endpoints with sample request/responses.
*   **Screenshot Evidence:**
    *   Swagger UI (`/docs`).
    *   Successful Ingestion.
    *   Text Query Result.
    *   Table Query Result.
    *   Image Query Result (summary chunks).
    *   Health Endpoint showing indexed count.

## 6. Evaluation Criteria
*   **RAG Accuracy:** Factual correctness, grounding, and reference accuracy.
*   **Code Quality:** Modularity, Pydantic models, error handling.
*   **Documentation:** Completeness of README and screenshots.
*   **Originality:** Unique problem statement and domain application.

## 7. Submission
*   **Format:** Public GitHub repository URL.
*   **Deadline:** 14 calendar days from issue.
*   **Integrity:** Natural development history (incremental commits) is expected.
