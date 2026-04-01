import json
import os
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent


def load_config() -> dict:
    """Load configuration from repo root first, then fall back to src/."""
    for config_file in (BASE_DIR / "config.json", SRC_DIR / "config.json"):
        if config_file.exists():
            with open(config_file, "r") as f:
                return json.load(f)
    return {}


# Load configuration
_config = load_config()


def get_setting(name: str, default: Optional[str] = None) -> Optional[str]:
    """Prefer environment variables, then config.json, then the provided default."""
    return os.getenv(name) or _config.get(name, default)


# Configuration constants
AWS_REGION: str = get_setting("AWS_REGION", "eu-central-1")
MODEL_ID: str = get_setting("MODEL_ID", "openai.gpt-oss-120b-1:0")
FIRECRAWL_API_URL: str = get_setting("FIRECRAWL_API_URL", "http://localhost:3002")

# Set AWS credentials from config if environment variables are not already present
if not os.getenv("AWS_ACCESS_KEY_ID") and _config.get("AWS_ACCESS_KEY_ID"):
    os.environ["AWS_ACCESS_KEY_ID"] = _config["AWS_ACCESS_KEY_ID"]

if not os.getenv("AWS_SECRET_ACCESS_KEY") and _config.get("AWS_SECRET_ACCESS_KEY"):
    os.environ["AWS_SECRET_ACCESS_KEY"] = _config["AWS_SECRET_ACCESS_KEY"]
