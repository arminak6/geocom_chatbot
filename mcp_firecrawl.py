import json
from typing import List, Optional, Dict
from urllib.parse import urlparse

from langchain_aws import ChatBedrock
from mcp import ClientSession, types

from utils import extract_json_from_model_output


# ====== MCP TOOL DISCOVERY ======

async def pick_firecrawl_tools(session: ClientSession) -> Dict[str, Optional[str]]:
    tools_result = await session.list_tools()
    tool_names = [t.name for t in tools_result.tools]

    scrape_tool = None
    crawl_tool = None
    map_tool = None

    for t in tools_result.tools:
        name = t.name.lower()
        if "scrape" in name and scrape_tool is None:
            scrape_tool = t.name
        if "crawl" in name and crawl_tool is None:
            crawl_tool = t.name
        if "map" in name and map_tool is None:
            map_tool = t.name

    if not scrape_tool:
        raise RuntimeError(f"No scrape-like MCP tool found in: {tool_names}")

    return {
        "scrape": scrape_tool,
        "crawl": crawl_tool,  
        "map": map_tool,     
    }


# ====== SINGLE PAGE SCRAPING ======

async def firecrawl_single_page_via_mcp(
    session: ClientSession,
    tool_name: str,
    url: str,
) -> str:
    arguments = {
        "url": url,
        "formats": ["markdown"],
        "onlyMainContent": False,
    }

    result = await session.call_tool(tool_name, arguments=arguments)

    if getattr(result, "isError", False):
        error_texts: List[str] = []
        for block in (result.content or []):
            if isinstance(block, types.TextContent):
                error_texts.append(block.text)
        msg = "\n".join(error_texts) if error_texts else "Unknown Firecrawl MCP error"
        raise RuntimeError(f"Firecrawl MCP error: {msg}")

    texts: List[str] = []
    for block in (result.content or []):
        if isinstance(block, types.TextContent):
            texts.append(str(block.text))

    md = "\n\n".join(texts).strip()
    return md


# ====== MULTI-PAGE CRAWLING ======

async def firecrawl_crawl_via_mcp(
    session: ClientSession,
    crawl_tool_name: str,
    url: str,
    max_depth: int = 2,
    max_pages: int = 20,
) -> str:

    if not crawl_tool_name:
        raise RuntimeError("Crawl tool not available in MCP.")

    args = {
        "url": url,
        "maxDepth": max_depth,
        "limit": max_pages,
    }

    result = await session.call_tool(crawl_tool_name, arguments=args)

    if getattr(result, "isError", False):
        error_texts: List[str] = []
        for block in (result.content or []):
            if isinstance(block, types.TextContent):
                error_texts.append(block.text)
        msg = "\n".join(error_texts) if error_texts else "Unknown Firecrawl MCP crawl error"
        raise RuntimeError(f"Firecrawl MCP crawl error: {msg}")

    md_chunks: List[str] = []

    for block in (result.content or []):
        if not isinstance(block, types.TextContent):
            continue
        text = block.text.strip()
        if not text:
            continue

        try:
            data = json.loads(text)
            if isinstance(data, dict) and "markdown" in data:
                md_chunks.append(data["markdown"])
            else:
                md_chunks.append(text)
        except json.JSONDecodeError:
            md_chunks.append(text)

    full_md = "\n\n---\n\n".join(md_chunks).strip()
    return full_md


# ====== SUBPAGE MAPPING & URL SCORING ======

async def firecrawl_map_subpages_via_mcp(
    session: ClientSession,
    map_tool_name: Optional[str],
    base_url: str,
    user_question: str,           
    llm: Optional[ChatBedrock],  
    max_links: int = 10,
) -> List[str]:

    if not map_tool_name:
        print("ℹ️ No Firecrawl 'map' tool available; skipping subpage discovery.")
        return []

    parsed_base = urlparse(base_url)
    base_domain = parsed_base.netloc.lower()

    raw_limit = max_links * 5

    args = {
        "url": base_url,
        "maxDepth": 2,
        "limit": raw_limit,
    }

    result = await session.call_tool(map_tool_name, arguments=args)

    if getattr(result, "isError", False):
        error_texts: List[str] = []
        for block in (result.content or []):
            if isinstance(block, types.TextContent):
                error_texts.append(block.text)
        msg = "\n".join(error_texts) if error_texts else "Unknown Firecrawl MCP map error"
        raise RuntimeError(f"Firecrawl MCP map error: {msg}")

    # Collect candidate URLs 
    candidates: List[str] = []
    seen = set()

    def add_candidate(candidate: str):
        nonlocal candidates, seen, base_domain, base_url, parsed_base
        if not candidate:
            return
        candidate = candidate.strip()
        if candidate.startswith("//"):
            candidate = parsed_base.scheme + ":" + candidate
        if candidate.startswith("/"):
            candidate = f"{parsed_base.scheme}://{parsed_base.netloc}{candidate}"

        parsed = urlparse(candidate)
        if not parsed.scheme.startswith("http"):
            return
        if parsed.netloc.lower() != base_domain:
            return
        if candidate == base_url:
            return
        if candidate in seen:
            return
        seen.add(candidate)
        candidates.append(candidate)

    for block in (result.content or []):
        if not isinstance(block, types.TextContent):
            continue
        text = block.text.strip()
        if not text:
            continue

        try:
            data = json.loads(text)
            links = (
                data.get("links")
                or data.get("data", {}).get("links")
                or data.get("data")
            )
            if isinstance(links, list):
                for link in links:
                    if isinstance(link, str):
                        add_candidate(link)
                    elif isinstance(link, dict):
                        candidate = (
                            link.get("url")
                            or link.get("href")
                            or link.get("link")
                        )
                        add_candidate(candidate)
        except json.JSONDecodeError:
            continue

    if not candidates:
        print("ℹ️ Firecrawl 'map' returned no same-domain links.")
        return []

    print(f"\n🗺️ Map API found {len(candidates)} candidate URLs:")
    for idx, url in enumerate(candidates, 1):
        print(f"  {idx}. {url}")
    print()

    # Use LLM to score URLs based on relevance to user question
    if llm is not None:
        url_scores = await score_urls_with_llm(llm, user_question, candidates)
    else:
        # Fallback: assign equal scores if no LLM available
        url_scores = {u: 5.0 for u in candidates}

    print("🔢 LLM-based URL scores:", url_scores)
    scored: List[tuple] = []
    for idx, u in enumerate(candidates):
        s = float(url_scores.get(u, 0.0))
        scored.append((s, idx, u))

    scored.sort(key=lambda x: (-x[0], x[1]))

    top = scored[:max_links]
    selected_urls = [u for (s, idx, u) in top]

    print("🔍 Deep dive: ranked subpage candidates (score, url):")
    for s, idx, u in top:
        print(f"  score={s:.2f}  url={u}")

    return selected_urls


# ====== LLM URL SCORING ======

async def score_urls_with_llm(
    llm: ChatBedrock,
    user_question: str,
    urls: List[str],
    chunk_size: int = 10,  
) -> Dict[str, float]:

    if not urls:
        return {}

    system_prompt = """
You are a URL ranking assistant.

You MUST respond in valid JSON ONLY, with no explanations, no prose, no markdown, and NO special tokens.

DO NOT use any control tokens like <|constrain|>, <|thinking|>, or similar.
DO NOT use <reasoning> tags.
Output ONLY the JSON object, nothing else.

Output format (exactly this shape):

{
  "scores": {
    "https://example.com/page1": 7.5,
    "https://example.com/page2": 3.0
  }
}

Rules:
- The first non-whitespace character must be '{'
- The last non-whitespace character must be '}'
- No extra keys
- No text before or after the JSON
- Do NOT wrap the JSON in backticks
- Do NOT include comments
- Assign each URL a relevance score from 0.0 to 10.0 based on how likely it is to contain information relevant to the user's question
- Think quickly and directly output the JSON
""".strip()

    all_scores: Dict[str, float] = {u: 0.0 for u in urls}

    for i in range(0, len(urls), chunk_size):
        chunk = urls[i:i + chunk_size]
        urls_text = "\n".join(f"- {u}" for u in chunk)

        user_prompt = (
            f"User question:\n{user_question}\n\n"
            f"Here is the list of URLs (only score these):\n{urls_text}\n\n"
            f"Return ONLY the JSON object with 'scores' as described, "
            f"and ONLY for the URLs in this list."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        result = await llm.ainvoke(messages)
        raw = getattr(result, "content", str(result))
        if isinstance(raw, list):
            raw_text = " ".join(str(x) for x in raw)
        else:
            raw_text = str(raw)

        print("📥 Raw LLM URL scoring output (chunk):")
        print(raw_text)

        cleaned = extract_json_from_model_output(raw_text)
        print("🧹 Cleaned JSON candidate for URL scoring (chunk):")
        print(repr(cleaned))

        if not cleaned or not cleaned.strip():
            print(f"⚠️ LLM URL scoring returned empty output for chunk (len={len(chunk)}). Assigning neutral score 5.0.")
            for u in chunk:
                all_scores[u] = 5.0
            continue

        try:
            data = json.loads(cleaned)
            scores = data.get("scores", {})
            if not scores:
                print(f"⚠️ No scores in LLM response for chunk. Assigning neutral score 5.0.")
                for u in chunk:
                    all_scores[u] = 5.0
            else:
                for u in chunk:
                    if u in scores:
                        all_scores[u] = float(scores[u])
                    else:
                        # Missing score for this URL, assign neutral
                        all_scores[u] = 5.0
                print("✅ Parsed URL scores for chunk:", {u: all_scores[u] for u in chunk})
        except Exception as e:
            print(f"⚠️ LLM URL scoring failed for chunk (len={len(chunk)}): {e}")
            for u in chunk:
                all_scores[u] = 5.0

    print("✅ Final merged URL scores:", all_scores)
    return all_scores
