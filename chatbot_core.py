import asyncio
import json
import re
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse

from langchain_aws import ChatBedrock
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

from config import AWS_REGION, MODEL_ID, FIRECRAWL_API_URL
from utils import extract_json_from_model_output, strip_reasoning_tags, extract_url_from_text


# ====== SYSTEM PROMPTS ======

PLANNER_SYSTEM_PROMPT = """
You are a decision-making assistant that chooses whether to use a web scraping tool.

You have access to a tool called "firecrawl", which can scrape or crawl websites.
A separate system will actually run the tool; YOU ONLY DECIDE WHETHER TO USE IT.

You must respond in valid JSON ONLY, with no extra text, no comments, no markdown.
Do NOT include <reasoning> tags, explanations, or any text before or after the JSON.
If you want to think step-by-step, do it internally; the final output MUST be pure JSON.

Output format (exactly one of these):

1) If you need to use Firecrawl:

{
  "action": "call_firecrawl",
  "reason": "brief explanation",
  "url": "https://example.com",
  "mode": "single_page"
}

Where "mode" is either:
- "single_page" -> use when the question only requires the content of ONE specific page (e.g. the exact URL the user gave).
- "crawl" -> use when the question likely needs information spread across MULTIPLE pages (e.g. big company sites, documentation, blogs, etc.).

2) If you can answer directly without scraping (or use cached content if available):

{
  "action": "answer_direct",
  "reason": "brief explanation"
}

3) If cached content exists and is sufficient:

{
  "action": "use_cache",
  "reason": "brief explanation"
}

Rules:
- Never add extra keys.
- Never wrap JSON in backticks.
- "url" must be absolute (start with http:// or https://) if you use call_firecrawl.
- Prefer using the tool when the question clearly depends on website content.
- If cached content is mentioned and seems relevant to the current question, you can choose "use_cache".
- If the new question seems to be about a different topic than cached content, choose "call_firecrawl" or "answer_direct".
"""

ANSWER_SYSTEM_PROMPT = """
- Only change language if the user explicitly asks for another language.

You are a helpful AI assistant.

You are given website content (scraped from the user's requested URL) and a user question.

Rules:

1. Treat the website content as the PRIMARY source of truth for facts about that company or site.
2. You ARE allowed to use general business and world knowledge to interpret what the content implies.
3. If the website does not explicitly state the answer but strongly suggests it, you may give a PROBABLE answer.
4. In such cases, be honest about uncertainty. Use phrases like:
   - "The website does not say this explicitly, but it suggests that..."
   - "It is likely that..."

5. If the website contains no relevant information, simply explain what the site *does* and *does not* say.
   - Do NOT suggest next actions, steps, recommendations, or guidance.

6. Formatting:
   - Use **Markdown** for all formatting (lists, bold words, etc.).
   - **Do NOT** use HTML tags (like <br>, <p>, <span>).
   - Be concise and easy to read.

7. Deep Analysis / Multiple Pages:
   - If you are provided with content from multiple pages, integrate the information into a cohesive answer.
   - Identify conflicting or complementary details found on subpages (e.g., "The homepage mentions X, but the contact page clarifies Y").
   - When content is labeled with source URLs (e.g., "--- Source: https://example.com ---"), 
     pay attention to which sources you reference in your answer.

8. Scoring / Likelihood:
   - **ONLY** provide a "Likelihood" score if, and only if, the user explicitly asks for an assessment of probability or "Are they looking for X?".
   - If the question is fact-seeking (e.g., "What is the address?", "Who is the CEO?"), just answer the question directly.

9. Source Citation (IMPORTANT):
   - After your answer, on a new line, add: SOURCES_USED: [url1, url2, url3]
   - Only list URLs that you ACTUALLY referenced or used in formulating your answer.
   - Do NOT list all available sources - only the ones you used.
   - If you only used one source, list only that one.
   - Use the exact URLs as provided in the source labels.
   - Format example: SOURCES_USED: [https://example.com, https://example.com/about]

Always answer clearly and only based on website content plus reasonable inference.
"""


# ====== MCP HELPER FUNCTIONS ======

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

    if llm is not None:
        llm_scores = await score_urls_with_llm(llm, user_question, candidates)
    else:
        llm_scores = {u: 0.0 for u in candidates}

    heuristic_scores = {u: heuristic_score_url(u, user_question) for u in candidates}

    url_scores = {}
    for u in candidates:
        llm_score = llm_scores.get(u, 0.0)
        h_score = heuristic_scores.get(u, 0.0)
        combined = (0.7 * llm_score) + (0.3 * h_score)
        url_scores[u] = combined

    print("🔢 Combined URL scores (70% LLM + 30% heuristic):", url_scores)
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


# ====== LLM FUNCTIONS ======

async def planner_decide_action(
    llm: ChatBedrock,
    user_message: str,
    cached_url: Optional[str] = None,
) -> Dict[str, Any]:

    if cached_url:
        user_prompt = (
            f"{user_message}\n\n"
            f"Note: We have cached content from: {cached_url}\n"
            f"If this question is related to that cached website, you can choose 'use_cache'. "
            f"Otherwise, decide based on the question."
        )
    else:
        user_prompt = user_message
    
    messages = [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    result = await llm.ainvoke(messages)
    raw = getattr(result, "content", str(result))
    if isinstance(raw, list):
        raw_text = " ".join(str(x) for x in raw)
    else:
        raw_text = str(raw)

    raw_text = raw_text.strip()
    cleaned = extract_json_from_model_output(raw_text)
    print(f"🧠 Planner cleaned output: {cleaned}")

    try:
        data = json.loads(cleaned)
        if not isinstance(data, dict):
            raise ValueError("Planner JSON is not an object")
    except Exception as e:
        print(f"⚠ Failed to parse planner JSON ({e}). Falling back to answer_direct.")
        return {
            "action": "answer_direct",
            "reason": "Planner output was not valid JSON; answer directly.",
        }

    action = data.get("action")
    if action == "call_firecrawl":
        mode = data.get("mode", "single_page")
        data["mode"] = mode

    if action not in {"call_firecrawl", "answer_direct", "use_cache"}:
        print(f"⚠️ Unknown planner action '{action}'. Falling back to answer_direct.")
        return {
            "action": "answer_direct",
            "reason": "Planner chose an unknown action; answer directly.",
        }
    return data


async def llm_answer(
    llm: ChatBedrock,
    user_message: str,
    website_content: Optional[str] = None,
    used_firecrawl: bool = False,
    history: Optional[list] = None,
    use_history: bool = False,
    source_urls: Optional[List[str]] = None,
) -> str:

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
    ]

    if use_history and history:
        for msg in history[-4:]:
            role = msg.get("role")
            content = msg.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

    if website_content:
        HEAD = 100000
        TAIL = 50000

        if len(website_content) > HEAD + TAIL:
            content_for_ai = (
                website_content[:HEAD]
                + "\n\n---\n\n"
                + website_content[-TAIL:]
            )
        else:
            content_for_ai = website_content
        user_prompt = (
            f"The user asked:\n{user_message}\n\n"
            f"I have scraped website content relevant to this question.\n\n"
            f"Here is the website content:\n\n"
            f"{content_for_ai}\n\n"
            f"Please answer the user's question using this content as the primary evidence, "
            f"following your system instructions about inference, uncertainty, and next steps."
        )
    else:
        user_prompt = user_message

    messages.append({"role": "user", "content": user_prompt})

    # LLM call
    result = await llm.ainvoke(messages)
    answer = getattr(result, "content", str(result))
    if isinstance(answer, list):
        answer_text = " ".join(str(x) for x in answer)
    else:
        answer_text = str(answer)

    answer_text = strip_reasoning_tags(answer_text)

    if used_firecrawl and source_urls:
        cited_sources = []
        
        sources_pattern = r'SOURCES_USED:\s*\[(.*?)\]'
        match = re.search(sources_pattern, answer_text, re.IGNORECASE | re.DOTALL)
        
        if match:
            sources_str = match.group(1)
            potential_urls = [url.strip().strip('"').strip("'") for url in sources_str.split(',')]
            
            for url in potential_urls:
                url = url.strip()
                if url and url in source_urls:
                    cited_sources.append(url)
            
            answer_text = re.sub(sources_pattern, '', answer_text, flags=re.IGNORECASE | re.DOTALL).strip()
            
            print(f"✅ Parsed {len(cited_sources)} cited sources from LLM response")
        else:
            print("⚠️ No SOURCES_USED citation found in LLM response, using all available sources")
        
        sources_to_display = cited_sources if cited_sources else source_urls
        
        answer_text += "\n\n---\n**📚 Sources:**\n"
        for idx, url in enumerate(sources_to_display, 1):
            answer_text += f"{idx}. {url}\n"

    return answer_text


# ====== URL SCORING ======

def heuristic_score_url(url: str, user_question: str) -> float:
    """
    Score URL relevance using heuristics.
    
    Args:
        url: URL to score
        user_question: User's question
        
    Returns:
        Relevance score between 0 and 10
    """
    url_lower = url.lower()
    question_lower = user_question.lower()
    
    high_relevance = [
        'about', 'team', 'people', 'management', 'leadership', 'staff',
        'employees', 'organization', 'who-we-are', 'chi-siamo', 'quienes-somos',
        'company', 'contact', 'contacts', 'address', 'location', 'careers',
        'jobs', 'work', 'join', 'offices', 'product', 'service', 'offering','technology', 'manufacturing',
    ]
    
    medium_relevance = [
         'solution', 
        'production', 'facility', 'plant', 'sustainability', 'quality', 'innovation' , 'news'
    ]
    
    low_relevance = [
        'blog', 'news', 'article', 'post', 'timeline', 'history', 'archive'
    ]
    
    score = 2.0  # Base score
    
    for keyword in high_relevance:
        if keyword in url_lower:
            score += 5.0
            break
    
    for keyword in medium_relevance:
        if keyword in url_lower:
            score += 2.5
            break
    
    for keyword in low_relevance:
        if keyword in url_lower:
            score -= 1.0
            break

    question_words = [w for w in question_lower.split() if len(w) > 3]
    for word in question_words[:5]:  
        if word in url_lower:
            score += 1.5
            break
    
    return max(0.0, min(10.0, score))


async def score_urls_with_llm(
    llm: ChatBedrock,
    user_question: str,
    urls: List[str],
    chunk_size: int = 10,  
) -> Dict[str, float]:
    """
    Ask the LLM to assign a relevance score 0–10 to each URL.
    
    Args:
        llm: ChatBedrock instance
        user_question: User's question
        urls: List of URLs to score
        chunk_size: Number of URLs to score per LLM call
        
    Returns:
        Dictionary mapping URLs to relevance scores
    """
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
            print(f"⚠️ LLM URL scoring returned empty output for chunk (len={len(chunk)}). Using heuristic scores.")
            for u in chunk:
                all_scores[u] = heuristic_score_url(u, user_question)
            continue

        try:
            data = json.loads(cleaned)
            scores = data.get("scores", {})
            if not scores:
                print(f"⚠️ No scores in LLM response for chunk. Using heuristic scores.")
                for u in chunk:
                    all_scores[u] = heuristic_score_url(u, user_question)
            else:
                for u in chunk:
                    if u in scores:
                        all_scores[u] = float(scores[u])
                    else:
                        all_scores[u] = heuristic_score_url(u, user_question)
                print("✅ Parsed URL scores for chunk:", {u: all_scores[u] for u in chunk})
        except Exception as e:
            print(f"⚠️ LLM URL scoring failed for chunk (len={len(chunk)}): {e}")
            for u in chunk:
                all_scores[u] = heuristic_score_url(u, user_question)

    print("✅ Final merged URL scores:", all_scores)
    return all_scores


# ====== MAIN CHAT LOGIC ======

async def chat_once(
    user_message: str,
    history: list,
    use_history: bool,
    session_state: dict
) -> str:

    if not FIRECRAWL_API_URL:
        raise RuntimeError("FIRECRAWL_API_URL must be set for self-hosted Firecrawl.")

    incoming_url = extract_url_from_text(user_message)
    force_scrape = incoming_url is not None

    last_url = session_state.get("last_url")
    last_md = session_state.get("last_site_markdown")

    session_state["last_user_question"] = user_message
    session_state["last_deep_done"] = False
    print(f"💬 Updated last_user_question to: '{user_message}'")

    if incoming_url and last_url and incoming_url != last_url:
        return (
            f"You're currently in a session for:\n\n- **{last_url}**\n\n"
            "To ask about a different website, please click **Reset (new website)** in the sidebar."
        )

    if not incoming_url and last_url and last_md:
        pass  

    if incoming_url and last_url == incoming_url and last_md:
        session_state["last_used_firecrawl"] = True

        return await llm_answer(
            llm=ChatBedrock(
                model_id=MODEL_ID,
                region_name=AWS_REGION,
                model_kwargs={
                    "temperature": 0.3,
                    "max_completion_tokens": 512,
                    "top_p": 0.9,
                    "reasoning_effort": "medium",
                },
            ),
            user_message=user_message,
            website_content=last_md,
            used_firecrawl=True,
            history=history,
            use_history=use_history,
            source_urls=[last_url] if last_url else None,
        )

    llm = ChatBedrock(
        model_id=MODEL_ID,
        region_name=AWS_REGION,
        model_kwargs={
            "temperature": 0.3,
            "max_completion_tokens": 512,
            "top_p": 0.9,
            "reasoning_effort": "medium",
        },
    )

    env = {"FIRECRAWL_API_URL": FIRECRAWL_API_URL}
    server_params = StdioServerParameters(command="firecrawl-mcp", args=[], env=env)

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as mcp_session:
            await mcp_session.initialize()

            tools = await pick_firecrawl_tools(mcp_session)
            scrape_tool_name = tools["scrape"]
            crawl_tool_name = tools.get("crawl")

            planner_decision = await planner_decide_action(
                llm, 
                user_message, 
                cached_url=last_url
            ) or {}
            mode = planner_decision.get("mode", "single_page")
            if mode not in ("single_page", "crawl"):
                mode = "single_page"

            if force_scrape:
                action = "call_firecrawl"
                url = incoming_url
            else:
                action = planner_decision.get("action", "answer_direct")
                url = planner_decision.get("url")
                url = extract_url_from_text(url) if url else None

            if action == "use_cache" and last_md:
                session_state["last_used_firecrawl"] = True
                return await llm_answer(
                    llm,
                    user_message,
                    website_content=last_md,
                    used_firecrawl=True,
                    history=history,
                    use_history=use_history,
                    source_urls=[last_url] if last_url else None,
                )
            
            if action != "call_firecrawl" or not url:
                session_state["last_used_firecrawl"] = False
                return await llm_answer(
                    llm,
                    user_message,
                    website_content=None,
                    used_firecrawl=False,
                    history=history,
                    use_history=use_history,
                )

            site_markdown = None
            for attempt in range(2):
                try:
                    if mode == "crawl" and crawl_tool_name:
                        site_markdown = await firecrawl_crawl_via_mcp(
                            mcp_session,
                            crawl_tool_name,
                            url,
                            max_depth=3,
                            max_pages=20,
                        )
                    else:
                        site_markdown = await firecrawl_single_page_via_mcp(
                            mcp_session,
                            scrape_tool_name,
                            url,
                        )
                    break
                except Exception as e:
                    print(f"❌ Firecrawl error (attempt {attempt + 1}/2): {e}")

            if site_markdown is None:
                session_state["last_used_firecrawl"] = False
                fallback = await llm_answer(
                    llm,
                    user_message,
                    website_content=None,
                    used_firecrawl=False,
                    history=history,
                    use_history=use_history,
                )
                return (
                    "I couldn't fetch the website content (Firecrawl failed). "
                    "This answer may be incomplete.\n\n"
                    + fallback
                )

            session_state["last_used_firecrawl"] = True

            if session_state.get("last_url") != url:
                session_state["last_url"] = url
                session_state["last_base_markdown"] = site_markdown
                session_state["last_site_markdown"] = site_markdown
                session_state["last_deep_done"] = False
                print(f"🆕 New website detected: {url} - saved fresh base content")
            else:
                existing_enriched = session_state.get("last_site_markdown")
                if existing_enriched and len(existing_enriched) > len(site_markdown):
                    print(f"💎 Preserving enriched content ({len(existing_enriched)} chars vs {len(site_markdown)} chars)")
                else:
                    session_state["last_site_markdown"] = site_markdown
                    if not session_state.get("last_base_markdown"):
                        session_state["last_base_markdown"] = site_markdown
                    print(f"📄 Updated site content (no enrichment lost)")

            return await llm_answer(
                llm,
                user_message,
                website_content=session_state["last_site_markdown"], 
                used_firecrawl=True,
                history=history,
                use_history=use_history,
                source_urls=[url] if url else None,
            )


async def deep_dive_once(
    history: list,
    use_history: bool,
    session_state: dict
) -> str:

    base_url = session_state.get("last_url")
    base_md = session_state.get("last_base_markdown")    
    user_question = session_state.get("last_user_question")

    print(f"🔍 Deep dive using question: '{user_question}'")  
    
    if not base_url or not base_md or not user_question:
        return "I cannot go deeper because there is no previous website context stored."

    if not FIRECRAWL_API_URL:
        raise RuntimeError("FIRECRAWL_API_URL must be set for self-hosted Firecrawl.")

    llm = ChatBedrock(
        model_id=MODEL_ID,
        region_name=AWS_REGION,
        model_kwargs={
            "temperature": 0.3,
            "max_completion_tokens": 1024,
            "top_p": 0.9,
            "reasoning_effort": "medium",
        },
    )

    env = {
        "FIRECRAWL_API_URL": FIRECRAWL_API_URL
    }

    server_params = StdioServerParameters(
        command="firecrawl-mcp",
        args=[],
        env=env,
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as mcp_session:
            await mcp_session.initialize()
            tools = await pick_firecrawl_tools(mcp_session)
            scrape_tool_name = tools["scrape"]
            map_tool_name = tools.get("map")

            try:
                subpage_urls = await firecrawl_map_subpages_via_mcp(
                    mcp_session,
                    map_tool_name,
                    base_url,
                    user_question=user_question,  
                    llm=llm,                       
                    max_links=10,
                )

            except Exception as e:
                print(f"⚠️ Firecrawl map error during deep dive: {e}")
                subpage_urls = []

            sub_markdowns: List[str] = []
            scraped_urls: List[str] = []  # Track successfully scraped URLs
            for u in subpage_urls:
                try:
                    print(f"🌐 Scraping subpage: {u}")
                    md = await firecrawl_single_page_via_mcp(
                        mcp_session,
                        scrape_tool_name,
                        u,
                    )
                    if md:
                        # Label the content with its source URL
                        labeled_md = f"--- Source: {u} ---\n\n{md}\n\n"
                        sub_markdowns.append(labeled_md)
                        scraped_urls.append(u)  # Track this URL
                except Exception as e:
                    print(f"⚠️ Error scraping subpage {u}: {e}")
                    continue

            # Label base content with its URL too
            labeled_base_md = f"--- Source: {base_url} ---\n\n{base_md}\n\n"
            all_md_chunks = [labeled_base_md] + sub_markdowns
            combined_markdown = "\n---\n\n".join(all_md_chunks)

            session_state["last_site_markdown"] = combined_markdown
            session_state["last_deep_done"] = True

            # Include base URL + all successfully scraped subpage URLs
            all_source_urls = [base_url] + scraped_urls

            answer = await llm_answer(
                llm,
                user_message=user_question,          
                website_content=combined_markdown,    
                used_firecrawl=True,
                history=history,
                use_history=use_history,
                source_urls=all_source_urls,
            )

            answer += "\n\n_(This is a deeper analysis using additional pages from the same site.)_"

            return answer
