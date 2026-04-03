# AutoDiag: Comprehensive Testing Guide

This guide provides step-by-step instructions to verify the **AutoDiag** RAG system after cloning the repository. Following these steps ensures the environment is correctly configured and the system is performing optimally.

## 1. Environment Setup
Before running the system, ensure you have a clean Python environment.

```bash
# 1. Clone the repository
git clone https://github.com/Rishi-Pokhrel/DirectML-ESG-RAG.git
cd DirectML-ESG-RAG

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install required dependencies
pip install -r requirements.txt
```

## 2. Model Initialization
AutoDiag uses a local 4-bit quantized LLM (**Qwen2.5-0.5B-Instruct-GGUF**) to stay within strict RAM limits. This file is ignored by Git due to its size and must be downloaded locally.

```bash
# Download the ~469MB model file to data/models/
python3 scripts/setup_model.py
```

## 3. Configuration & Secrets
Copy the environment template and add your Google Gemini API Key (required for vision/multimodal features).

```bash
cp .env.example .env
# Open .env and add your GOOGLE_API_KEY=your_actual_key
```

## 4. Running the Live System
Start the FastAPI server. The system uses lazy loading, so the LLM and Vector Store will initialize upon the first request to save startup RAM.

```bash
export PYTHONPATH=$PYTHONPATH:.
python3 main.py
```

## 5. Adaptive Resource Management (The Guardrail)
AutoDiag features a **Hardware Guardrail System** that detects available RAM before processing the first query. This ensures the system remains stable on the target 520MB hardware.

### A. One-Click Mode (RAM > 1GB)
On standard computers or high-spec codespaces, the system automatically detects ample memory. When you type your first query, it will:
1. Initialize the embedding model.
2. Ingest all technical manuals from `sample_documents/`.
3. Flush the ingestion memory.
4. Load the 0.5B LLM and generate your answer.
**Observation:** You only need to click the **Query** button once.

### B. Safety Mode (RAM < 1GB)
On ultra-low resource devices (like 250MB-520MB hardware), the system prevents "Memory Overlap"—a condition where loading the embedding model and the LLM simultaneously would cause an OS-level crash.
1. When you click Query, the AI will post an **ALERT** message.
2. The user is guided to click the **"Re-Ingest"** button first.
3. This manually prepares the database and **flushes the RAM buffers** before the Query engine is allowed to load.
**Observation:** This ensures the "Hacker-Proof" stability requested for field diagnostics.

## 6. Verification Steps

### A. Technical Query Test (Text-Based)
Input the following query:
> *"What are the main components of a planetary gear set?"*
*   **Expected Result:** A structured list including the sun gear, planetary gears, and ring gear.

### B. Exhaustive Diagnostic Test (Precision RAG)
Input the following query:
> *"How do you troubleshoot a slipping automatic transmission?"*
*   **Expected Result:** A long, bolded, step-by-step guide (10+ points) retrieved from high-density micro-chunks.

## 7. Troubleshooting
*   **ModuleNotFoundError:** Ensure you ran `export PYTHONPATH=$PYTHONPATH:.` before starting the server.
*   **Empty Results:** If no answer is provided, ensure you have clicked "Re-Ingest" or that the "Docs" count in the header is > 0.
