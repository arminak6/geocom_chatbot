import asyncio
import streamlit as st

from chatbot_core import chat_once, deep_dive_once



def reset_conversation():
    """Reset all session state variables."""
    st.session_state.history = []
    st.session_state.last_used_firecrawl = False
    st.session_state.last_url = None
    st.session_state.last_site_markdown = None
    st.session_state.last_base_markdown = None
    st.session_state.last_user_question = None
    st.session_state.last_deep_done = False
    st.session_state.last_deep_question = None


def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    if "history" not in st.session_state:
        st.session_state.history = []
    
    if "last_used_firecrawl" not in st.session_state:
        st.session_state.last_used_firecrawl = False
        
    if "last_url" not in st.session_state:
        st.session_state.last_url = None
        
    if "last_site_markdown" not in st.session_state:
        st.session_state.last_site_markdown = None
        
    if "last_base_markdown" not in st.session_state:
        st.session_state.last_base_markdown = None
        
    if "last_user_question" not in st.session_state:
        st.session_state.last_user_question = None
        
    if "last_deep_done" not in st.session_state:
        st.session_state.last_deep_done = False
        
    if "last_deep_question" not in st.session_state:
        st.session_state.last_deep_question = None


# ====== UI LAYOUT ======

def render_sidebar():
    """Render sidebar with session controls."""
    with st.sidebar:
        st.subheader("Session")
        if st.button("Reset (new website)", use_container_width=True):
            reset_conversation()
            st.rerun()


def render_chat_history():
    """Render existing chat history."""
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def render_deep_dive_button():
    """Render deep dive button if conditions are met."""
    if (
        st.session_state.get("last_url") is not None
        and st.session_state.get("last_site_markdown") is not None
        and not st.session_state.get("last_deep_done", False)
    ):
        if st.button("Go deeper on this site (scan more pages)"):
            with st.chat_message("assistant"):
                with st.spinner("Scanning more pages on this site..."):
                    deep_answer = asyncio.run(
                        deep_dive_once(
                            history=st.session_state.history,
                            use_history=True,
                            session_state=st.session_state,
                        )
                    )
                    st.markdown(deep_answer)
            st.session_state.history.append({"role": "assistant", "content": deep_answer})


# ====== MAIN APP ======

def main():
    """Main application entry point."""
    st.set_page_config(page_title="Praeciso Chatbot", page_icon="🔥")
    
    st.title("🔥 Praeciso Chatbot")
    st.caption("Ask about any website. The bot decides when to scrape via Firecrawl MCP. You can optionally go deeper on a site.")
    
    initialize_session_state()
    
    render_sidebar()
    
    render_chat_history()
    
    user_message = st.chat_input("Ask something about a website")
    
    if user_message:
        with st.chat_message("user"):
            st.markdown(user_message)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = asyncio.run(
                    chat_once(
                        user_message,
                        history=st.session_state.history,
                        use_history=True,
                        session_state=st.session_state,
                    )
                )
                st.markdown(answer)
        
        st.session_state.history.append({"role": "user", "content": user_message})
        st.session_state.history.append({"role": "assistant", "content": answer})
    
    render_deep_dive_button()


if __name__ == "__main__":
    main()
