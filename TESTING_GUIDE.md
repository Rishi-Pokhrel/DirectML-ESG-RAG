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

## 5. Verification Steps

### A. Web Interface Test
1. Open `http://localhost:8000` in your browser.
2. Click the **"Re-Ingest"** button. 
3. **Observation:** Check the terminal logs. You should see the system performing "Precision Micro-Chunking" on the PDFs in `sample_documents/`.
4. **Result:** The "Docs" count in the top-right header should increase (expected: ~660 micro-chunks).

### B. Technical Query Test (Text-Based)
Input the following query:
> *"What are the main components of a planetary gear set?"*
*   **Expected Result:** A structured list including the sun gear, planetary gears, and ring gear.

### C. Exhaustive Diagnostic Test (Precision RAG)
Input the following query:
> *"How do you troubleshoot a slipping automatic transmission?"*
*   **Expected Result:** A long, bolded, step-by-step guide (10+ points) retrieved from high-density micro-chunks.

### D. Resource Monitoring
Check the header of the Web UI while querying.
*   **Observation:** RAM usage should spike during generation but stay stable due to `mmap` and explicit garbage collection.

## 6. Troubleshooting
*   **ModuleNotFoundError:** Ensure you ran `export PYTHONPATH=$PYTHONPATH:.` before starting the server.
*   **OutOfMemory:** If the system crashes, check `logs/run_log.json`. Ensure `n_threads` in `config/settings.json` is set to 2 for low-RAM devices.
*   **Empty Results:** Ensure you clicked "Re-Ingest" at least once to build the local Vector Database (`data/vector_db`).
