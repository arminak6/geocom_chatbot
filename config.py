import json
import os
from pathlib import Path
from typing import Optional


def load_config() -> dict:
    """Load configuration from config.json if it exists."""
    config_file = Path(__file__).parent / "config.json"
    
    if config_file.exists():
        with open(config_file, 'r') as f:
            return json.load(f)
    return {}


# Load configuration
_config = load_config()

# Configuration constants
AWS_REGION: str = _config.get("AWS_REGION", "eu-central-1")
MODEL_ID: str = _config.get("MODEL_ID", "openai.gpt-oss-20b-1:0")
FIRECRAWL_API_URL: str = _config.get("FIRECRAWL_API_URL", "http://localhost:3002")

# Set AWS credentials from config if provided
if _config.get("AWS_ACCESS_KEY_ID"):
    os.environ["AWS_ACCESS_KEY_ID"] = _config["AWS_ACCESS_KEY_ID"]
    
if _config.get("AWS_SECRET_ACCESS_KEY"):
    os.environ["AWS_SECRET_ACCESS_KEY"] = _config["AWS_SECRET_ACCESS_KEY"]
