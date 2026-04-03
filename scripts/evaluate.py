import os
import json
import time
import psutil
from src.ingestion.processor import DocumentProcessor
from src.retrieval.engine import RAGEngine
from src.utils.logger import setup_logger

logger = setup_logger("evaluation")

def evaluate_system():
    # 1. Measure Ingestion Performance
    print("--- Stage 1: Ingestion ---")
    start_time = time.time()
    processor = DocumentProcessor()
    
    raw_files = [
        "data/raw/AutoTrans_9781284122039_samplech11.pdf",
        "data/raw/Crawfords_Auto_Repair_Guide.pdf",
        "data/raw/Passenger Car Drive Axle Technology.pdf"
    ]
    
    for pdf in raw_files:
        print(f"Processing {pdf}...")
        processor.process_pdf(pdf)
    
    ingestion_time = time.time() - start_time
    print(f"Ingestion completed in {ingestion_time:.2f} seconds.")

    # 2. Measure Query Performance & RAM
    print("\n--- Stage 2: RAG Query Evaluation ---")
    engine = RAGEngine()
    
    test_queries = [
        "What are the main components of a planetary gear set?",
        "How do you troubleshoot a slipping automatic transmission?",
        "What is the purpose of a differential in a drive axle?"
    ]
    
    results = []
    process = psutil.Process(os.getpid())
    
    for query in test_queries:
        print(f"Querying: {query}")
        q_start = time.time()
        response = engine.answer_query(query)
        q_time = time.time() - q_start
        
        mem_usage = process.memory_info().rss / (1024 * 1024)
        print(f"Response: {response['answer'][:100]}...")
        print(f"Time: {q_time:.2f}s | RAM: {mem_usage:.2f} MB")
        
        results.append({
            "query": query,
            "answer": response["answer"],
            "sources": response["sources"],
            "latency_sec": q_time,
            "ram_usage_mb": mem_usage
        })

    # 3. Save Evaluation Report
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": "Qwen2.5-0.5B-Instruct-GGUF (Q4_K_M)",
        "hardware_stats": {
            "total_ingestion_time_sec": ingestion_time,
            "avg_query_latency_sec": sum(r["latency_sec"] for r in results) / len(results),
            "peak_ram_usage_mb": max(r["ram_usage_mb"] for r in results)
        },
        "results": results
    }
    
    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/evaluation_report.json", "w") as f:
        json.dump(report, f, indent=4)
    
    print(f"\nEvaluation complete. Report saved to data/processed/evaluation_report.json")

if __name__ == "__main__":
    evaluate_system()
