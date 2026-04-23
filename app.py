import base64
import csv
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st

# ── Must be the FIRST Streamlit call ─────────────────────────────────────────
st.set_page_config(
    page_title="RAG Support Assistant",
    page_icon="log.jpg",
    layout="wide",  # Changed to wide for a more modern feel
)

from graph import create_graph, _reload_vectorstore   # noqa: E402
from ingest import main as ingest_main                 # noqa: E402

# ── Professional UI Styling ──────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;600;700&display=swap');

    /* ── Base ── */
    :root {
        --primary-color: #2563eb;
        --bg-color: #f8fafc;
        --sidebar-bg: #ffffff;
        --text-main: #1e293b;
        --text-muted: #64748b;
        --border-color: #e2e8f0;
        --card-bg: #ffffff;
        --accent-color: #3b82f6;
    }

    .stApp {
        background-color: var(--bg-color) !important;
        color: var(--text-main) !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background-color: var(--sidebar-bg) !important;
        border-right: 1px solid var(--border-color) !important;
    }
    section[data-testid="stSidebar"] .stMarkdown hr {
        border-color: var(--border-color) !important;
    }
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3 {
        font-family: 'Outfit', sans-serif !important;
        color: var(--text-main) !important;
    }

    /* ── Main content area ── */
    .main .block-container {
        padding-top: 3rem !important;
        max-width: 1000px !important;
    }

    /* ── Headings ── */
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif !important;
        color: var(--text-main) !important;
        font-weight: 700 !important;
    }
    
    .stMarkdown p, .stMarkdown li {
        color: var(--text-main) !important;
        line-height: 1.6 !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background-color: transparent !important;
        border-bottom: 1px solid var(--border-color) !important;
        gap: 24px !important;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        color: var(--text-muted) !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        padding: 10px 4px !important;
        border: none !important;
    }
    .stTabs [aria-selected="true"] {
        color: var(--primary-color) !important;
        border-bottom: 2px solid var(--primary-color) !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: var(--primary-color) !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        width: 100%;
        border-radius: 10px !important;
        background-color: var(--primary-color) !important;
        color: white !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        border: none !important;
        padding: 10px 20px !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
    }
    .stButton > button:hover {
        background-color: var(--accent-color) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05) !important;
    }
    
    /* Secondary buttons (like Rebuild) */
    [data-testid="stSidebar"] .stButton > button {
        background-color: transparent !important;
        color: var(--text-main) !important;
        border: 1px solid var(--border-color) !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background-color: #f1f5f9 !important;
    }

    /* ── Chat messages ── */
    .stChatMessage {
        background-color: var(--card-bg) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 16px !important;
        padding: 1.5rem !important;
        margin-bottom: 1rem !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02) !important;
    }
    [data-testid="stChatMessageContent"] p {
        color: var(--text-main) !important;
    }

    /* ── Chat input ── */
    .stChatInput {
        padding-bottom: 2rem !important;
    }
    .stChatInput textarea {
        background-color: var(--card-bg) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        color: var(--text-main) !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05) !important;
    }
    .stChatInput button {
        background-color: var(--primary-color) !important;
        color: white !important;
        border-radius: 8px !important;
    }

    /* ── Metrics ── */
    [data-testid="stMetric"] {
        background-color: var(--card-bg) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        padding: 16px !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03) !important;
    }
    [data-testid="stMetricValue"] {
        color: var(--primary-color) !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] {
        color: var(--text-muted) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        font-weight: 600 !important;
    }

    /* ── Expanders ── */
    .stExpander {
        background-color: var(--card-bg) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        overflow: hidden !important;
    }
    .stExpander summary {
        font-weight: 600 !important;
        color: var(--text-main) !important;
    }

    /* ── Alerts ── */
    .stAlert {
        border-radius: 12px !important;
        border: none !important;
    }

    /* ── Progress bar ── */
    .stProgress > div > div {
        background-color: var(--primary-color) !important;
    }

    /* ── Custom Classes ── */
    .main-title {
        font-size: 2.5rem !important;
        margin-bottom: 0.5rem !important;
        background: linear-gradient(90deg, #1e293b 0%, #2563eb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)
with open("log.jpg", "rb") as f:
    data = base64.b64encode(f.read()).decode()
    
st.markdown(f"""
    <h1 style="display:flex; align-items:center;">
        <img src="data:image/jpeg;base64,{data}" width="40" style="margin-right:10px;">
        RAG Support Assistant
    </h1>
""", unsafe_allow_html=True)

st.markdown(
    "<p style='color: #64748b; font-size: 1.1rem; margin-bottom: 2rem;'>"
    "Intelligent customer support powered by your knowledge base. "
    "Complex queries are automatically routed to human experts."
    "</p>", 
    unsafe_allow_html=True
)


# ── Graph (cached per Streamlit session) ──────────────────────────────────────
@st.cache_resource
def load_graph():
    return create_graph()

graph_app = load_graph()


# ── Session state defaults ────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "thread_id":        str(uuid.uuid4()),
        "case_id":          str(uuid.uuid4()),
        "messages":         [],
        "escalated_queries": [],
        # Per-query flags — reset after each completed turn
        "hitl_pending":     False,   # True while a query awaits human review
        "current_config":   None,
        "draft_answer":     "",
        "current_docs":     [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# Add a welcome message on first load
if not st.session_state.messages:
    st.session_state.messages.append({
        "role":    "assistant",
        "content": (
            "Hi! 👋 I'm your support assistant. "
            "Ask me anything about our services, policies, or FAQs."
        ),
    })


# ── Helpers ───────────────────────────────────────────────────────────────────

def _new_thread():
    """Generate a fresh thread_id so LangGraph starts a clean run."""
    st.session_state.thread_id = str(uuid.uuid4())


def run_graph(query: str):
    """
    Invoke LangGraph for a new user query.
    Returns (final_answer | None, state_values).
    None means the graph paused at human_review.
    """
    # Each query uses its own thread so old state never bleeds in.
    _new_thread()
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    st.session_state.current_config = config

    # Stream the graph to completion (or first interrupt)
    for _ in graph_app.stream({"query": query}, config):
        pass

    state = graph_app.get_state(config)

    if state.next:
        # Paused before human_review
        st.session_state.hitl_pending  = True
        st.session_state.draft_answer  = state.values.get("draft_answer", "")
        st.session_state.current_docs  = state.values.get("retrieved_context", [])

        st.session_state.escalated_queries.append({
            "query":      query,
            "docs":       state.values.get("retrieved_context", []),
            "draft":      state.values.get("draft_answer", ""),
            "confidence": state.values.get("confidence_score", 0.0),
            "intent":     state.values.get("intent", "unknown"),
            "config":     config,
            "resolved":   False,
        })
        return None, state.values

    # Completed without interrupt
    st.session_state.hitl_pending = False
    return state.values.get("final_answer", "Sorry, I couldn't process that."), state.values


def log_case(case_id, user_query, bot_answer, confidence, intent, escalated):
    """Append a row to support_cases.csv."""
    file_path = "support_cases.csv"
    columns   = ["timestamp", "case_id", "user_query", "bot_answer",
                  "confidence", "intent", "escalated"]
    write_header = not Path(file_path).exists()
    with open(file_path, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(columns)
        w.writerow([
            datetime.now().isoformat(), case_id, user_query, bot_answer,
            f"{confidence:.2f}", intent, escalated,
        ])


def _resume_graph(item: dict, approved_answer: str):
    """Resume a paused graph thread after human edits the answer."""
    graph_app.update_state(
        item["config"],
        {"draft_answer": approved_answer},
        as_node="human_review",
    )
    for _ in graph_app.stream(None, item["config"]):
        pass
    final_state = graph_app.get_state(item["config"])
    return final_state.values.get("final_answer", approved_answer)


# ── Sidebar: PDF upload + re-ingest + metrics ─────────────────────────────────
with st.sidebar:
    st.title("📁 Knowledge Base")
    st.markdown("Manage the documents that power your assistant.")
    
    uploaded = st.file_uploader("Upload PDF", type="pdf")
    if uploaded:
        with open("knowledge_base.pdf", "wb") as f:
            f.write(uploaded.read())
        st.success("✅ PDF saved successfully")

    if st.button("🔄 Rebuild Index"):
        with st.spinner("Processing documents..."):
            ok = ingest_main()
            if ok:
                _reload_vectorstore()
                st.success("✅ Index rebuilt.")
            else:
                st.error("❌ Ingest failed.")

    st.markdown("---")
    st.header("📊 Performance")
    
    total_q = len([m for m in st.session_state.messages if m["role"] == "user"])
    escalated_n = len(st.session_state.escalated_queries)
    
    st.metric("Total Queries", total_q)
    st.metric("Escalations", escalated_n)
    if total_q:
        st.metric("Escalation Rate", f"{escalated_n / total_q:.0%}")


# ── Main tabs ─────────────────────────────────────────────────────────────────
tab_chat, tab_admin = st.tabs(["💬 Customer Chat", "🛠️ HITL Admin"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 – Customer Chat
# ═══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    # Render full conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # If we're waiting for a human agent, block new input
    if st.session_state.hitl_pending:
        st.warning(
            "⏳ **Escalated to Expert:** Your query requires specialized attention. "
            "A human agent is reviewing it now. You'll see the response here shortly."
        )
    else:
        if prompt := st.chat_input("How can I help you today?"):
            # Show user bubble immediately
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.spinner("Searching knowledge base..."):
                final_answer, state_vals = run_graph(prompt)

            if final_answer is None:
                # Graph paused — tell user and stop
                st.rerun()
            else:
                # Successful direct answer
                confidence = state_vals.get("confidence_score", 0.0)
                intent     = state_vals.get("intent", "unknown")

                st.session_state.messages.append({"role": "assistant", "content": final_answer})
                log_case(
                    st.session_state.case_id, prompt, final_answer,
                    confidence, intent, False,
                )

                with st.chat_message("assistant"):
                    st.markdown(final_answer)

                # Confidence and Sources in a clean layout
                c1, c2 = st.columns([1, 1])
                with c1:
                    if confidence > 0:
                        label = "High" if confidence >= 0.7 else ("Medium" if confidence >= 0.4 else "Low")
                        color = "green" if confidence >= 0.7 else ("orange" if confidence >= 0.4 else "red")
                        st.markdown(f"**Confidence:** :{color}[{label} ({confidence:.0%})]")
                
                with c2:
                    docs = state_vals.get("retrieved_context", [])
                    if docs:
                        with st.expander("📚 View Sources"):
                            for i, doc in enumerate(docs, 1):
                                page = doc.metadata.get("page", "N/A")
                                st.markdown(f"**Source {i}** (Page {page})")
                                st.caption(doc.page_content[:300] + "...")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 – HITL Admin
# ═══════════════════════════════════════════════════════════════════════════════
with tab_admin:
    st.header("🛠️ Human-in-the-Loop Admin")
    st.markdown("Review and approve responses for escalated customer queries.")

    pending = [q for q in st.session_state.escalated_queries if not q.get("resolved")]

    if not pending:
        st.info("✨ All caught up! No pending escalations.")
    else:
        for idx, item in enumerate(pending):
            master_idx = st.session_state.escalated_queries.index(item)

            with st.container():
                st.markdown(f"### Escalation #{idx + 1}")
                st.info(f"**User Query:** {item['query']}")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("AI Confidence", f"{item['confidence']:.0%}")
                col2.metric("Detected Intent", item["intent"])
                
                if item["docs"]:
                    with st.expander("📖 Reference Context"):
                        for i, doc in enumerate(item["docs"], 1):
                            page = doc.metadata.get("page", "N/A")
                            st.markdown(f"**Chunk {i}** (Page {page})")
                            st.caption(doc.page_content)

                edited = st.text_area(
                    "Draft Response",
                    value=item["draft"],
                    height=200,
                    key=f"edit_{master_idx}",
                    help="You can refine the AI's draft before sending it to the customer."
                )

                btn_col1, btn_col2, _ = st.columns([1, 1, 2])

                with btn_col1:
                    if st.button("✅ Approve & Send", key=f"approve_{master_idx}"):
                        final_answer = _resume_graph(item, edited)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": final_answer}
                        )
                        log_case(
                            st.session_state.case_id, item["query"],
                            final_answer, item["confidence"], item["intent"], True,
                        )
                        st.session_state.escalated_queries[master_idx]["resolved"] = True
                        st.session_state.hitl_pending = False
                        st.success("Response sent!")
                        st.rerun()

                with btn_col2:
                    if st.button("❌ Reject", key=f"skip_{master_idx}"):
                        rejection = "I'm sorry, I'm unable to answer this question at this time. Please contact us directly for assistance."
                        final_answer = _resume_graph(item, rejection)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": final_answer}
                        )
                        log_case(
                            st.session_state.case_id, item["query"],
                            final_answer, item["confidence"], item["intent"], True,
                        )
                        st.session_state.escalated_queries[master_idx]["resolved"] = True
                        st.session_state.hitl_pending = False
                        st.rerun()
                
                st.markdown("---")
