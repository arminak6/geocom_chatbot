import asyncio
import json
import os
import re
import shutil
from typing import List, Optional, Dict, Any
from urllib.parse import urljoin, urlparse

from langchain_aws import ChatBedrock
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

from config import AWS_REGION, MODEL_ID, FIRECRAWL_API_URL
from utils import (
    canonical_site_host,
    extract_json_from_model_output,
    extract_url_from_text,
    normalize_session_url,
    site_url_variants,
    strip_reasoning_tags,
)
from mcp_firecrawl import (
    pick_firecrawl_tools,
    firecrawl_single_page_via_mcp,
    firecrawl_crawl_via_mcp,
    firecrawl_map_subpages_with_metadata_via_mcp,
    score_urls_with_llm,
)


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

DEEP_DIVE_MAX_LINKS = 10
DEEP_DIVE_GLOBAL_RATIO = 0.3
DEEP_DIVE_MAP_LIMIT = DEEP_DIVE_MAX_LINKS * 5


def build_firecrawl_server_params() -> StdioServerParameters:
    """Prefer an installed Firecrawl MCP binary and fall back to npx when needed."""
    env = dict(os.environ)
    env["FIRECRAWL_API_URL"] = FIRECRAWL_API_URL

    if shutil.which("firecrawl-mcp"):
        return StdioServerParameters(command="firecrawl-mcp", args=[], env=env)

    return StdioServerParameters(command="npx", args=["-y", "firecrawl-mcp"], env=env)


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
    print(f"[planner] cleaned output: {cleaned}")

    try:
        data = json.loads(cleaned)
        if not isinstance(data, dict):
            raise ValueError("Planner JSON is not an object")
    except Exception as e:
        print(f"[warning] Failed to parse planner JSON ({e}). Falling back to answer_direct.")
        return {
            "action": "answer_direct",
            "reason": "Planner output was not valid JSON; answer directly.",
        }

    action = data.get("action")
    if action == "call_firecrawl":
        mode = data.get("mode", "single_page")
        data["mode"] = mode

    if action not in {"call_firecrawl", "answer_direct", "use_cache"}:
        print(f"[warning] Unknown planner action '{action}'. Falling back to answer_direct.")
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
        HEAD = 200000
        TAIL = 100000

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
            
            print(f"[sources] Parsed {len(cited_sources)} cited sources from LLM response")
        else:
            print("[warning] No SOURCES_USED citation found in LLM response, using all available sources")
        
        sources_to_display = cited_sources if cited_sources else source_urls
        
        answer_text += "\n\n---\n**Sources:**\n"
        for idx, url in enumerate(sources_to_display, 1):
            answer_text += f"{idx}. {url}\n"

    return answer_text


def _normalize_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""

    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return url

    return parsed._replace(fragment="").geturl()


def _same_site(left: Optional[str], right: Optional[str]) -> bool:
    return bool(left and right and canonical_site_host(left) == canonical_site_host(right))


def _same_session_url(left: Optional[str], right: Optional[str]) -> bool:
    return bool(left and right and normalize_session_url(left) == normalize_session_url(right))


def _dedupe_urls(urls: List[str]) -> List[str]:
    seen = set()
    deduped: List[str] = []

    for url in urls:
        normalized = _normalize_url(url)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)

    return deduped


def _create_deep_dive_state(base_url: str) -> Dict[str, Any]:
    return {
        "base_url": base_url,
        "step": 0,
        "scraped_urls": [],
        "candidate_unseen_urls": [],
        "global_candidate_urls": [],
        "last_round_top_urls": [],
        "scraped_page_markdowns": {},
        "scraped_url_order": [],
    }


def _get_or_init_deep_dive_state(session_state: dict, base_url: str) -> Dict[str, Any]:
    state = session_state.get("deep_dive_state")
    if not isinstance(state, dict) or state.get("base_url") != base_url:
        state = _create_deep_dive_state(base_url)
        session_state["deep_dive_state"] = state

    state.setdefault("step", 0)
    state.setdefault("scraped_urls", [])
    state.setdefault("candidate_unseen_urls", [])
    state.setdefault("global_candidate_urls", [])
    state.setdefault("last_round_top_urls", [])
    state.setdefault("scraped_page_markdowns", {})
    state.setdefault("scraped_url_order", [])

    if not isinstance(state["scraped_page_markdowns"], dict):
        state["scraped_page_markdowns"] = {}

    return state


def _sort_urls_by_score(urls: List[str], url_scores: Dict[str, float]) -> List[str]:
    scored = [
        (float(url_scores.get(url, 0.0)), idx, url)
        for idx, url in enumerate(urls)
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [url for _, _, url in scored]


def _build_combined_markdown(base_url: str, base_md: str, deep_dive_state: Dict[str, Any]) -> str:
    labeled_chunks = [f"--- Source: {base_url} ---\n\n{base_md}\n\n"]
    page_markdowns = deep_dive_state.get("scraped_page_markdowns", {})

    for url in deep_dive_state.get("scraped_url_order", []):
        md = page_markdowns.get(url)
        if not md:
            continue
        labeled_chunks.append(f"--- Source: {url} ---\n\n{md}\n\n")

    return "\n---\n\n".join(labeled_chunks)


def _collect_source_urls(base_url: str, deep_dive_state: Dict[str, Any]) -> List[str]:
    return _dedupe_urls([base_url] + deep_dive_state.get("scraped_url_order", []))


async def _map_same_domain_subpages(
    mcp_session: ClientSession,
    map_tool_name: Optional[str],
    seed_url: str,
    base_url: str,
    limit: int = DEEP_DIVE_MAP_LIMIT,
) -> List[str]:
    if not map_tool_name:
        print("[info] No Firecrawl 'map' tool available; skipping subpage discovery.")
        return []

    base_domain = canonical_site_host(base_url)
    base_session_url = normalize_session_url(base_url)
    seed_session_url = normalize_session_url(seed_url)
    last_error: Optional[RuntimeError] = None

    for map_seed_url in site_url_variants(seed_url):
        args = {
            "url": map_seed_url,
            "maxDepth": 2,
            "limit": limit,
        }

        result = await mcp_session.call_tool(map_tool_name, arguments=args)

        if getattr(result, "isError", False):
            error_texts: List[str] = []
            for block in (result.content or []):
                if isinstance(block, types.TextContent):
                    error_texts.append(block.text)
            msg = "\n".join(error_texts) if error_texts else "Unknown Firecrawl MCP map error"
            last_error = RuntimeError(f"Firecrawl MCP map error for {map_seed_url}: {msg}")
            continue

        candidates: List[str] = []
        seen = set()

        def add_candidate(candidate: Optional[str]) -> None:
            if not candidate:
                return

            normalized = _normalize_url(urljoin(map_seed_url, candidate.strip()))
            if not normalized or normalized in seen:
                return

            parsed = urlparse(normalized)
            if not parsed.scheme.startswith("http"):
                return
            if canonical_site_host(normalized) != base_domain:
                return
            if normalize_session_url(normalized) in {base_session_url, seed_session_url}:
                return

            seen.add(normalized)
            candidates.append(normalized)

        for block in (result.content or []):
            if not isinstance(block, types.TextContent):
                continue

            text = block.text.strip()
            if not text:
                continue

            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                continue

            payload = data.get("data")
            payload_links = payload.get("links") if isinstance(payload, dict) else payload
            links = data.get("links") or payload_links
            if not isinstance(links, list):
                continue

            for link in links:
                if isinstance(link, str):
                    add_candidate(link)
                elif isinstance(link, dict):
                    add_candidate(
                        link.get("url")
                        or link.get("href")
                        or link.get("link")
                    )

        if candidates:
            if map_seed_url != seed_url:
                print(f"[map] Retried subpage discovery with alternate host: {map_seed_url}")
            print(f"[map] Found {len(candidates)} same-domain links from {map_seed_url}")
            return candidates

    if last_error is not None:
        raise last_error

    print(f"[map] Found 0 same-domain links from {seed_url}")
    return []


async def _select_deep_dive_urls_for_round(
    llm: ChatBedrock,
    mcp_session: ClientSession,
    map_tool_name: Optional[str],
    base_url: str,
    user_question: str,
    deep_dive_state: Dict[str, Any],
) -> Dict[str, Any]:
    step = int(deep_dive_state.get("step", 0) or 0)
    scraped_set = set(_dedupe_urls(deep_dive_state.get("scraped_urls", [])))

    if step == 0:
        initial_ranking = await firecrawl_map_subpages_with_metadata_via_mcp(
            mcp_session,
            map_tool_name,
            base_url,
            user_question=user_question,
            llm=llm,
            max_links=DEEP_DIVE_MAX_LINKS,
        )

        selected_urls = initial_ranking.get("selected_urls", [])
        selected_urls = [
            url
            for url in _dedupe_urls(selected_urls)
            if url not in scraped_set
        ]

        initial_candidate_urls = [
            url
            for url in _dedupe_urls(initial_ranking.get("candidate_urls", []))
            if url not in scraped_set
        ]

        print(f"[deep-dive] Round 1 selected {len(selected_urls)} URLs")
        return {
            "selected_urls": selected_urls,
            "global_candidate_urls": initial_candidate_urls,
            "neighbor_candidate_urls": [],
        }

    # Preserve an all-history unseen frontier so future rounds can still
    # select strong links discovered earlier, not only the last round.
    historical_unseen_urls = deep_dive_state.get("candidate_unseen_urls", [])
    discovered_global_urls = await _map_same_domain_subpages(
        mcp_session,
        map_tool_name,
        base_url,
        base_url,
    )
    global_candidate_urls = [
        url
        for url in _dedupe_urls(
            historical_unseen_urls
            + deep_dive_state.get("global_candidate_urls", [])
            + discovered_global_urls
        )
        if url not in scraped_set
    ]

    neighbor_seed_urls = _dedupe_urls(deep_dive_state.get("last_round_top_urls", []))
    neighbor_discovered_urls: List[str] = []
    for seed_url in neighbor_seed_urls:
        try:
            neighbor_discovered_urls.extend(
                await _map_same_domain_subpages(
                    mcp_session,
                    map_tool_name,
                    seed_url,
                    base_url,
                )
            )
        except Exception as e:
            print(f"[warning] Firecrawl map error for neighbor seed {seed_url}: {e}")

    neighbor_candidate_urls = [
        url
        for url in _dedupe_urls(neighbor_discovered_urls)
        if url not in scraped_set
    ]

    neighbor_candidate_set = set(neighbor_candidate_urls)
    global_candidate_urls = [
        url
        for url in global_candidate_urls
        if url not in neighbor_candidate_set
    ]

    global_scores = (
        await score_urls_with_llm(llm, user_question, global_candidate_urls)
        if global_candidate_urls else {}
    )
    neighbor_scores = (
        await score_urls_with_llm(llm, user_question, neighbor_candidate_urls)
        if neighbor_candidate_urls else {}
    )

    ranked_global_urls = _sort_urls_by_score(global_candidate_urls, global_scores)
    ranked_neighbor_urls = _sort_urls_by_score(neighbor_candidate_urls, neighbor_scores)

    global_target = int(DEEP_DIVE_MAX_LINKS * DEEP_DIVE_GLOBAL_RATIO)
    neighbor_target = DEEP_DIVE_MAX_LINKS - global_target

    selected_urls = _dedupe_urls(
        ranked_neighbor_urls[:neighbor_target] + ranked_global_urls[:global_target]
    )

    if len(selected_urls) < DEEP_DIVE_MAX_LINKS:
        if len(ranked_neighbor_urls) < neighbor_target:
            preferred_fill = ranked_global_urls
        elif len(ranked_global_urls) < global_target:
            preferred_fill = ranked_neighbor_urls
        else:
            preferred_fill = ranked_neighbor_urls + ranked_global_urls

        for url in preferred_fill:
            if url in selected_urls:
                continue
            selected_urls.append(url)
            if len(selected_urls) >= DEEP_DIVE_MAX_LINKS:
                break

    if len(selected_urls) < DEEP_DIVE_MAX_LINKS:
        for url in ranked_global_urls + ranked_neighbor_urls:
            if url in selected_urls:
                continue
            selected_urls.append(url)
            if len(selected_urls) >= DEEP_DIVE_MAX_LINKS:
                break

    combined_scores = dict(global_scores)
    combined_scores.update(neighbor_scores)
    selected_urls = _sort_urls_by_score(selected_urls[:DEEP_DIVE_MAX_LINKS], combined_scores)

    print(
        f"[deep-dive] Round {step + 1}: "
        f"{len(ranked_global_urls)} global candidates, "
        f"{len(ranked_neighbor_urls)} neighbor candidates, "
        f"{len(selected_urls)} selected"
    )

    return {
        "selected_urls": selected_urls,
        "global_candidate_urls": global_candidate_urls,
        "neighbor_candidate_urls": neighbor_candidate_urls,
    }


# ====== MAIN CHAT LOGIC ======

async def chat_once(
    user_message: str,
    history: list,
    use_history: bool,
    session_state: dict
) -> str:

    if not FIRECRAWL_API_URL:
        raise RuntimeError("FIRECRAWL_API_URL must be set for self-hosted Firecrawl.")

    previous_user_question = (session_state.get("last_user_question") or "").strip()
    current_user_question = (user_message or "").strip()
    incoming_url = extract_url_from_text(user_message)
    force_scrape = incoming_url is not None

    last_url = session_state.get("last_url")
    last_md = session_state.get("last_site_markdown")
    last_source_urls = session_state.get("last_source_urls")
    same_site_as_last = _same_site(incoming_url, last_url)
    same_session_url_as_last = _same_session_url(incoming_url, last_url)

    session_state["last_user_question"] = user_message
    session_state["last_deep_done"] = False
    print(f"[chat] Updated last_user_question to: '{user_message}'")

    if (
        previous_user_question
        and previous_user_question != current_user_question
        and last_url
        and (incoming_url is None or same_site_as_last)
    ):
        session_state["deep_dive_state"] = _create_deep_dive_state(last_url)
        session_state["last_deep_question"] = None
        print("[chat] Reset deep-dive state for a new question on the same site")

    if incoming_url and last_url and not same_site_as_last:
        return (
            f"You're currently in a session for:\n\n- **{last_url}**\n\n"
            "To ask about a different website, please click **Reset (new website)** in the sidebar."
        )

    if not incoming_url and last_url and last_md:
        pass  

    if incoming_url and same_session_url_as_last and last_md:
        session_state["last_used_firecrawl"] = True

        return await llm_answer(
            llm=ChatBedrock(
                model_id=MODEL_ID,
                region_name=AWS_REGION,
                model_kwargs={
                    "temperature": 0.3,
                    "max_completion_tokens": 2048,
                    "top_p": 0.9,
                    "reasoning_effort": "medium",
                },
            ),
            user_message=user_message,
            website_content=last_md,
            used_firecrawl=True,
            history=history,
            use_history=use_history,
            source_urls=last_source_urls or ([last_url] if last_url else None),
        )

    llm = ChatBedrock(
        model_id=MODEL_ID,
        region_name=AWS_REGION,
        model_kwargs={
            "temperature": 0.3,
            "max_completion_tokens": 2048,
            "top_p": 0.9,
            "reasoning_effort": "medium",
        },
    )

    server_params = build_firecrawl_server_params()

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
                    source_urls=last_source_urls or ([last_url] if last_url else None),
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
                    print(f"[error] Firecrawl error (attempt {attempt + 1}/2): {e}")

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
                session_state["last_source_urls"] = [url] if url else None
                session_state["last_deep_done"] = False
                session_state["deep_dive_state"] = _create_deep_dive_state(url)
                print(f"[chat] New website detected: {url} - saved fresh base content")
            else:
                existing_enriched = session_state.get("last_site_markdown")
                if existing_enriched and len(existing_enriched) > len(site_markdown):
                    print(f"[chat] Preserving enriched content ({len(existing_enriched)} chars vs {len(site_markdown)} chars)")
                else:
                    session_state["last_site_markdown"] = site_markdown
                    session_state["last_source_urls"] = [url] if url else None
                    if not session_state.get("last_base_markdown"):
                        session_state["last_base_markdown"] = site_markdown
                    print("[chat] Updated site content (no enrichment lost)")

            return await llm_answer(
                llm,
                user_message,
                website_content=session_state["last_site_markdown"], 
                used_firecrawl=True,
                history=history,
                use_history=use_history,
                source_urls=session_state.get("last_source_urls") or ([url] if url else None),
            )


async def deep_dive_once(
    history: list,
    use_history: bool,
    session_state: dict
) -> str:

    base_url = session_state.get("last_url")
    base_md = session_state.get("last_base_markdown")    
    user_question = session_state.get("last_user_question")

    print(f"[deep-dive] Using question: '{user_question}'")
    
    if not base_url or not base_md or not user_question:
        return "I cannot go deeper because there is no previous website context stored."

    if not FIRECRAWL_API_URL:
        raise RuntimeError("FIRECRAWL_API_URL must be set for self-hosted Firecrawl.")

    llm = ChatBedrock(
        model_id=MODEL_ID,
        region_name=AWS_REGION,
        model_kwargs={
            "temperature": 0.3,
            "max_completion_tokens": 4096,
            "top_p": 0.9,
            "reasoning_effort": "medium",
        },
    )

    server_params = build_firecrawl_server_params()

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as mcp_session:
            await mcp_session.initialize()
            tools = await pick_firecrawl_tools(mcp_session)
            scrape_tool_name = tools["scrape"]
            map_tool_name = tools.get("map")

            deep_dive_state = _get_or_init_deep_dive_state(session_state, base_url)

            try:
                selection_plan = await _select_deep_dive_urls_for_round(
                    llm=llm,
                    mcp_session=mcp_session,
                    map_tool_name=map_tool_name,
                    base_url=base_url,
                    user_question=user_question,
                    deep_dive_state=deep_dive_state,
                )
            except Exception as e:
                print(f"[warning] Firecrawl map error during deep dive: {e}")
                selection_plan = {
                    "selected_urls": [],
                    "global_candidate_urls": deep_dive_state.get("global_candidate_urls", []),
                    "neighbor_candidate_urls": [],
                }

            selected_urls = selection_plan.get("selected_urls", [])
            if not selected_urls:
                session_state["last_site_markdown"] = _build_combined_markdown(base_url, base_md, deep_dive_state)
                session_state["last_source_urls"] = _collect_source_urls(base_url, deep_dive_state)
                session_state["last_deep_done"] = True
                return "I couldn't find any additional unseen subpages to analyze on this site."

            successful_scraped_urls: List[str] = []
            page_markdowns = deep_dive_state.get("scraped_page_markdowns", {})
            scraped_url_order = deep_dive_state.get("scraped_url_order", [])

            for u in selected_urls:
                try:
                    print(f"[scrape] Scraping subpage: {u}")
                    md = await firecrawl_single_page_via_mcp(
                        mcp_session,
                        scrape_tool_name,
                        u,
                    )
                    if not md:
                        continue

                    page_markdowns[u] = md
                    if u not in scraped_url_order:
                        scraped_url_order.append(u)
                    successful_scraped_urls.append(u)
                except Exception as e:
                    print(f"[warning] Error scraping subpage {u}: {e}")
                    continue

            deep_dive_state["scraped_page_markdowns"] = page_markdowns
            deep_dive_state["scraped_url_order"] = _dedupe_urls(scraped_url_order)
            deep_dive_state["scraped_urls"] = _dedupe_urls(
                deep_dive_state.get("scraped_urls", []) + successful_scraped_urls
            )
            deep_dive_state["last_round_top_urls"] = _dedupe_urls(selected_urls)
            deep_dive_state["step"] = int(deep_dive_state.get("step", 0) or 0) + 1

            scraped_set = set(deep_dive_state["scraped_urls"])
            remaining_global_candidates = [
                url
                for url in selection_plan.get("global_candidate_urls", [])
                if url not in scraped_set
            ]
            remaining_neighbor_candidates = [
                url
                for url in selection_plan.get("neighbor_candidate_urls", [])
                if url not in scraped_set
            ]
            deep_dive_state["global_candidate_urls"] = remaining_global_candidates
            deep_dive_state["candidate_unseen_urls"] = _dedupe_urls(
                remaining_global_candidates + remaining_neighbor_candidates
            )
            session_state["deep_dive_state"] = deep_dive_state

            combined_markdown = _build_combined_markdown(base_url, base_md, deep_dive_state)
            session_state["last_site_markdown"] = combined_markdown
            session_state["last_source_urls"] = _collect_source_urls(base_url, deep_dive_state)
            session_state["last_deep_done"] = True
            session_state["last_deep_question"] = user_question

            answer = await llm_answer(
                llm,
                user_message=user_question,
                website_content=combined_markdown,
                used_firecrawl=True,
                history=history,
                use_history=use_history,
                source_urls=session_state["last_source_urls"],
            )

            if successful_scraped_urls:
                answer += "\n\n_(This is a deeper analysis using additional pages from the same site.)_"
            else:
                answer += "\n\n_(A deeper round was attempted, but the newly selected pages could not be scraped successfully.)_"

            return answer


