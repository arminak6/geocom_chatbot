import re
from typing import List, Optional
from urllib.parse import urlparse


RE_REASONING_BLOCK = re.compile(r"<reasoning>.*?</reasoning>|<\|[^|]+\|>", re.DOTALL)

url_finder = re.compile(
    r"(?<!@)\b((?:https?://|www\.)[^\s]+|(?:[a-z0-9-]+\.)+[a-z]{2,}(?:/[^\s]*)?)",
    re.IGNORECASE,
)


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


def canonical_site_host(value: str) -> str:
    text = (value or "").strip()
    if not text:
        return ""

    parsed = urlparse(text if "://" in text else f"https://{text}")
    host = (parsed.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def normalize_session_url(url: str) -> str:
    text = (url or "").strip()
    if not text:
        return ""

    parsed = urlparse(text if "://" in text else f"https://{text}")
    host = canonical_site_host(text)
    if not host:
        return ""

    path = (parsed.path or "/").rstrip("/") or "/"
    query = f"?{parsed.query}" if parsed.query else ""
    return f"{host}{path}{query}"


def site_url_variants(url: str) -> List[str]:
    text = (url or "").strip()
    if not text:
        return []

    parsed = urlparse(text if "://" in text else f"https://{text}")
    scheme = parsed.scheme or "https"
    host = (parsed.hostname or "").lower()
    if not host:
        return []

    path = parsed.path or ""
    query = f"?{parsed.query}" if parsed.query else ""
    suffix = f"{path}{query}"

    candidate_hosts = [host]
    if host.startswith("www."):
        candidate_hosts.append(host[4:])
    else:
        candidate_hosts.append(f"www.{host}")

    variants: List[str] = []
    for candidate_host in candidate_hosts:
        candidate = f"{scheme}://{candidate_host}{suffix}"
        if candidate not in variants:
            variants.append(candidate)

    return variants


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