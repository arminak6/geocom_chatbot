import re
from typing import Optional


RE_REASONING_BLOCK = re.compile(r"<reasoning>.*?</reasoning>|<\|[^|]+\|>", re.DOTALL)

url_finder = re.compile(r"(https?://[^\s]+|www\.[^\s]+)", re.IGNORECASE)


def extract_json_from_model_output(raw_text: str) -> str:
    cleaned = RE_REASONING_BLOCK.sub("", raw_text).strip()

    if cleaned.lstrip().startswith("{"):
        return cleaned.strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and start < end:
        return cleaned[start: end + 1].strip()
    
    return cleaned


def strip_reasoning_tags(text: str) -> str:
    return RE_REASONING_BLOCK.sub("", text).strip()


def extract_url_from_text(text: str) -> Optional[str]:
    m = url_finder.search(text or "")
    if not m:
        return None

    raw_url = m.group(1).strip().rstrip('"\',;:!?)]}>/')
    
    # Normalize URL
    if raw_url.startswith("www."):
        return "https://" + raw_url
    if not raw_url.startswith("http://") and not raw_url.startswith("https://"):
        return "https://" + raw_url
    
    return raw_url
