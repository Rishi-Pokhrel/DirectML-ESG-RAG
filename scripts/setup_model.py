import os
import requests
from tqdm import tqdm
import json

def download_file(url: str, dest_path: str):
    """Downloads a file with progress bar."""
    if os.path.exists(dest_path):
        print(f"File already exists at {dest_path}")
        return

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    print(f"Downloading model to {dest_path}...")
    with open(dest_path, "wb") as f, tqdm(
        desc=dest_path,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)

def setup():
    # Load config
    with open("config/settings.json", "r") as f:
        config = json.load(f)["model"]

    # Hugging Face URL for Qwen2.5-0.5B-Instruct-GGUF Q4_K_M
    model_url = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"
    
    # Ensure data/models exists
    os.makedirs("data/models", exist_ok=True)
    
    # Download model
    download_file(model_url, config["local_path"])
    print("Setup complete. You can now run 'python main.py' to start the server.")

if __name__ == "__main__":
    setup()
