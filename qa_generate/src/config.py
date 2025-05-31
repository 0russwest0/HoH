import os
from pathlib import Path

MODE = "update" # "init" or "update"

# Time constants
OLD_MONTH = "2411"
NEW_MONTH = "2412"

# Model settings
MODEL_CONFIG = {
    "name": "Llama-3-1-70B-Instruct", # TODO: change to your own model
    "api_key": "<your_api_key>", # TODO: change to your own model server
    "base_url": "http://localhost:8000/v1/", # TODO: change to your own model server
    "temperature": 0.3,
    "max_tokens": 100
}

# Processing settings
BATCH_SIZE = 10
MAX_RETRIES = 3

# Path settings
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
INTERMEDIATE_DIR = DATA_DIR / "qa" / f"intermediate_results_{OLD_MONTH}01_{NEW_MONTH}01"

# Ensure directories exist
INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)

# File paths
DIFF_FILE = DATA_DIR / "diff" / f"diff_{OLD_MONTH}01_{NEW_MONTH}01_text.jsonl"
INPUT_QA_FILE = DATA_DIR / "qa" / f"all_qas_240601_{OLD_MONTH}01.json"
OUTPUT_QA_FILE = DATA_DIR / "qa" / f"all_qas_240601_{NEW_MONTH}01.json" 