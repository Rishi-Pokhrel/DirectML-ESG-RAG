import requests
import json
import time
import os

BASE_URL = "http://0.0.0.0:8000"

def test_health():
    """Verify the /health endpoint is online and reports correct metrics."""
    print("Testing /health...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        assert response.status_code == 200
        data = response.json()
        print(f"Health check passed! RAM: {data['memory_usage_mb']:.2f}MB, Indexed: {data['indexed_documents']}")
    except Exception as e:
        print(f"Health check FAILED: {str(e)}")

def test_ingest():
    """Verify the /ingest endpoint can process all documents in data/raw."""
    print("\nTesting /ingest...")
    try:
        response = requests.post(f"{BASE_URL}/ingest")
        assert response.status_code == 200
        data = response.json()
        print(f"Ingestion check passed! Processed files: {data['processed_files']}")
    except Exception as e:
        print(f"Ingestion check FAILED: {str(e)}")

def test_query():
    """Verify the /query endpoint provides a valid RAG answer with sources."""
    print("\nTesting /query...")
    query_data = {"text": "How do you troubleshoot a slipping automatic transmission?"}
    try:
        response = requests.post(f"{BASE_URL}/query", json=query_data)
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert len(data["sources"]) > 0
        print(f"Query check passed! Received a {len(data['answer'])} character answer.")
        print(f"Answer snippet: {data['answer'][:200]}...")
    except Exception as e:
        print(f"Query check FAILED: {str(e)}")

if __name__ == "__main__":
    # Wait for server to start if needed
    time.sleep(5)
    test_health()
    test_ingest()
    test_query()
