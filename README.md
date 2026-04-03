# AutoDiag: Precision Multimodal RAG for Automotive Diagnostics

## Problem Statement
In my experience working with automotive systems, particularly during diagnostic teardowns of automatic transmissions and drive axles, I've noticed a persistent, frustrating barrier: the shear weight and complexity of technical service manuals. These aren't just books; they are massive 300+ page PDFs filled with high-density text, exploded planetary gear views, and complex hydraulic flow charts. For an independent mechanic or a student in a high-pressure workshop, finding a specific torque spec or a "slipping transmission" diagnostic protocol is like looking for a needle in a haystack of PDFs.

The real problem isn't just the amount of data; it's how it's presented. When you're under a car, you don't need a search tool that gives you "100 matches for transmission." You need the specific, exhaustive steps for *that* symptom. Currently, critical diagnostic info is often "trapped" in diagrams or fine-print tables that standard keyword search simply ignores. I once spent over two hours skimming a manual just to find a single axle-shim specification that was buried in an engineering chart. This "search fatigue" leads to diagnostic errors and massive downtime.

Furthermore, most modern "AI assistants" are useless in a real garage. They require high-spec laptops with 16GB of RAM and a constant, fast internet connection. Most workshop tablets are low-spec, often with as little as 512MB of usable RAM. I wanted to build a system that actually works on those devices—something that stays offline, respects the hardware limits, and "understands" the multimodal nature of automotive manuals without needing a server rack.

**AutoDiag** is my solution to this. It's a precision-engineered RAG system designed specifically for the low-resource hardware we actually use in the field. By using "Micro-Chunking" to extract high-density technical snippets and running a 4-bit quantized model that fits in under 520MB of RAM, I've created a tool that provides step-by-step diagnostic protocols from my own library of automotive manuals. It doesn't just "summarize"—it retrieves the actual, granular technical steps needed on the shop floor.

## Architecture Overview
The system follows a modular "Precision RAG" pipeline:

```mermaid
graph TD
    subgraph Ingestion
        A[PDF Documents] --> B[pypdf Parser]
        B --> C[Precision Micro-Chunking]
        C --> D[SentenceTransformer]
        D --> E[(ChromaDB)]
    end
    subgraph Query_Pipeline
        F[User Query] --> G[Vector Store Query]
        E --> G
        G --> H[Context Snippets]
        H --> I[Qwen-0.5B LLM]
        I --> J[Technical Answer]
    end
    subgraph Monitoring
        K[FastAPI /health] --> L[psutil Metrics]
    end
```

1.  **Ingestion:** PDFs are parsed using `pypdf`. Technical text is split into high-density 500-character "Micro-Chunks" to ensure no diagnostic step is lost in large chunks. Images are identified for future multimodal synthesis via Gemini Vision.
2.  **Retrieval:** Uses `ChromaDB` (persistent) and `SentenceTransformers` (all-MiniLM-L6-v2) for CPU-efficient vector search. We retrieve the top 10 most relevant micro-chunks to maximize technical context density.
3.  **Inference:** Employs `Qwen2.5-0.5B-Instruct` in a 4-bit GGUF format via `llama-cpp-python`. The model uses `mmap` to keep the memory footprint below the 520MB target during generation.
4.  **Web Interface:** A sleek, dark-themed FastAPI portal for real-time querying and health monitoring.

## Technology Choices
- **LLM:** Qwen2.5-0.5B (4-bit). Chosen for its superior technical reasoning at an extremely small size (0.5GB), fitting the 520MB RAM target.
- **Vector Store:** ChromaDB. Lightweight, serverless, and supports persistent on-disk storage.
- **Parser:** pypdf. Low memory overhead compared to heavier OCR-based libraries.
- **Embedding:** all-MiniLM-L6-v2. The industry standard for high-speed, low-RAM CPU embeddings.

## Setup Instructions (For Accessors)
To run this project after cloning, follow these exact steps to ensure the reference model is initialized:

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd DirectML-ESG-RAG
   ```
2. **Create a Virtual Environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Download the 4-bit Quantized Reference Model:**
   Our script will download the required `Qwen2.5-0.5B-Instruct-GGUF` model (~469MB) to the `data/models/` folder.
   ```bash
   python3 scripts/setup_model.py
   ```
5. **Configure Environment Variables:**
   Rename `.env.example` to `.env` and add your `GOOGLE_API_KEY`.
   ```bash
   cp .env.example .env
   ```
6. **Launch the Application:**
   ```bash
   export PYTHONPATH=$PYTHONPATH:.
   python3 main.py
   ```
7. **Access the Interface:** Open `http://localhost:8000` in your browser.

## API Documentation
- `GET /`: Serves the web interface.
- `GET /health`: Returns RAM usage, document count, and status.
- `POST /ingest`: Triggers precision ingestion of all PDFs in `data/raw/`.
- `POST /query`: Accepts a JSON `{"text": "query"}` and returns an exhaustive technical answer with source citations.

## Screenshots
*(Screenshots to be placed in screenshots/ folder per checklist)*
- `Swagger UI`: Available at `/docs`.
- `Technical Query`: Demonstrating step-by-step diagnostic output.
- `Health Check`: Showing resource efficiency.

## Limitations & Future Work
- **Multimodal Synthesis:** Currently detects images; full integration with Gemini Vision for diagram-to-text synthesis is the next milestone.
- **Hardware Scale:** While optimized for 520MB, the Python runtime environment (libraries) adds overhead that could be further reduced by moving to a Rust or C++ backend.
