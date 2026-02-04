import asyncio
import json
import re
from typing import List, Optional, Dict, Any

from langchain_aws import ChatBedrock
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from config import AWS_REGION, MODEL_ID, FIRECRAWL_API_URL
from utils import extract_json_from_model_output, strip_reasoning_tags, extract_url_from_text
from mcp_firecrawl import (
    pick_firecrawl_tools,
    firecrawl_single_page_via_mcp,
    firecrawl_crawl_via_mcp,
    firecrawl_map_subpages_via_mcp,
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
            
            print(f"✅ Parsed {len(cited_sources)} cited sources from LLM response")
        else:
            print("⚠️ No SOURCES_USED citation found in LLM response, using all available sources")
        
        sources_to_display = cited_sources if cited_sources else source_urls
        
        answer_text += "\n\n---\n**📚 Sources:**\n"
        for idx, url in enumerate(sources_to_display, 1):
            answer_text += f"{idx}. {url}\n"

    return answer_text


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
            source_urls=[last_url] if last_url else None,
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
            "max_completion_tokens": 4096,
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