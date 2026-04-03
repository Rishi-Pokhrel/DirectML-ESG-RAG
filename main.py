import uvicorn
from src.api.routes import app
import json
import os

def main():
    # Load config
    with open("config/settings.json", "r") as f:
        config = json.load(f)["api"]
        
    print(f"Starting Automotive RAG Server on {config['host']}:{config['port']}...")
    uvicorn.run(app, host=config["host"], port=config["port"], workers=config["workers"])

if __name__ == "__main__":
    main()
