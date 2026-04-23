# graph.py
"""
LangGraph workflow for RAG Customer Support Assistant (Groq + ChromaDB).

Flow:  START → retrieve → classify_intent → generate
                                                  ↓              ↓
                                           human_review      finalize → END
                                                  ↓
                                             finalize → END

Escalation rules (BOTH must be true to escalate):
  • confidence_score < CONFIDENCE_THRESHOLD   (retrieved docs are weak)
  AND
  • intent is in SENSITIVE_INTENTS            (topic is risky)
OR
  • No docs found at all.

This means routine billing / refund / policy questions that are well-covered
in the knowledge base are answered directly.
"""

import os
from typing import TypedDict, Literal

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────
CHROMA_FOLDER      = "chroma_db"
EMBEDDING_MODEL    = "all-MiniLM-L6-v2"
GROQ_MODEL         = "llama-3.1-8b-instant"

# Escalate ONLY when confidence is below this AND intent is sensitive.
# Raise this value (e.g. 0.55) if you want more escalations.
CONFIDENCE_THRESHOLD = 0.35

# Only these intents are considered "sensitive" for escalation purposes.
SENSITIVE_INTENTS = {"refund", "policy", "billing"}

# Greetings / small-talk that bypass the vector search entirely
GREETINGS = {"hi", "hello", "hey", "hiya", "hi there", "good morning", "good afternoon"}
# ──────────────────────────────────────────────────────────────────────────────


class RAGState(TypedDict):
    query:             str
    retrieved_context: list
    draft_answer:      str
    final_answer:      str
    needs_human:       bool
    escalated:         bool
    confidence_score:  float
    avg_distance:      float
    intent:            str


# ── Lazy-initialised singletons (loaded once per process) ─────────────────────
_embeddings   = None
_vectorstore  = None
_llm          = None


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings


def _get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            persist_directory=CHROMA_FOLDER,
            embedding_function=_get_embeddings(),
        )
    return _vectorstore


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGroq(model=GROQ_MODEL, temperature=0)
    return _llm


def _reload_vectorstore():
    """Call this after re-ingesting so the graph picks up the new index."""
    global _vectorstore
    _vectorstore = Chroma(
        persist_directory=CHROMA_FOLDER,
        embedding_function=_get_embeddings(),
    )


# ── Node: retrieve ─────────────────────────────────────────────────────────────
def retrieve_node(state: RAGState) -> dict:
    print("--- NODE: retrieve ---")
    query = state["query"]

    # Short-circuit small talk — no retrieval needed
    if query.strip().lower() in GREETINGS or "thank" in query.lower():
        return {
            "retrieved_context": [],
            "needs_human":       False,
            "confidence_score":  1.0,
            "avg_distance":      0.0,
        }

    vs = _get_vectorstore()
    docs_and_scores = vs.similarity_search_with_score(query, k=4)

    docs, scores = [], []
    for doc, score in docs_and_scores:
        docs.append(doc)
        scores.append(score)
        print(f"  chunk distance={score:.3f}")

    if scores:
        avg_dist   = sum(scores) / len(scores)
        # Chroma returns L2 distance; convert to a 0-1 confidence score.
        # distance ~0  → confidence ~1   (very similar)
        # distance ~2  → confidence ~0   (very dissimilar)
        confidence = max(0.0, min(1.0, 1.0 - (avg_dist / 2.0)))
    else:
        avg_dist   = 2.0
        confidence = 0.0

    print(f"  {len(docs)} doc(s) | avg_dist={avg_dist:.3f} | confidence={confidence:.1%}")

    return {
        "retrieved_context": docs,
        "needs_human":       confidence < CONFIDENCE_THRESHOLD,
        "confidence_score":  confidence,
        "avg_distance":      avg_dist,
    }

# ── Node: classify_intent ──────────────────────────────────────────────────────
def classify_intent_node(state: RAGState) -> dict:
    print("--- NODE: classify_intent ---")
    query = state["query"]

    # Small-talk bypass
    if query.strip().lower() in GREETINGS or "thank" in query.lower():
        return {"intent": "general"}

    prompt = (
        "Classify the customer support query into EXACTLY ONE of these labels "
        "(output the label only, nothing else):\n"
        "billing | refund | technical | policy | general\n\n"
        "Rules:\n"
        "- Any mention of refund, money back, return payment → refund\n"
        "- Any mention of privacy, data, deletion, GDPR, policy, legal, terms → policy\n"
        "- Any mention of login, browser, API, error, bug, technical → technical\n"
        "- Any mention of invoice, charge, payment method, billing cycle → billing\n"
        "- Everything else → general\n\n"
        f"Query: {query}\n"
        "Label:"
    )

    raw = _get_llm().invoke(prompt).content.strip().lower()

    # Normalise — keep only the first recognised word
    intent = "general"
    for candidate in ("billing", "refund", "technical", "policy", "general"):
        if candidate in raw:
            intent = candidate
            break

    print(f"  intent={intent!r}  (raw LLM output: {raw!r})")
    return {"intent": intent}


# ── Node: generate ─────────────────────────────────────────────────────────────
def generate_node(state: RAGState) -> dict:
    print("--- NODE: generate ---")
    query      = state["query"]
    intent     = state.get("intent", "general")
    docs       = state.get("retrieved_context", [])
    needs_human = state.get("needs_human", False)

    # ── Small-talk shortcuts ──────────────────────────────────────────────────
    q_lower = query.strip().lower()
    if q_lower in GREETINGS:
        return {
            "draft_answer": (
                "Hi there! 👋 I'm your support assistant. "
                "Feel free to ask me anything about our services, policies, or FAQs."
            ),
            "escalated": False,
        }
    if "thank" in q_lower:
        return {
            "draft_answer": "You're very welcome! Let me know if there's anything else I can help with. 😊",
            "escalated": False,
        }

    # ── Escalation decision ───────────────────────────────────────────────────
    # Escalate ONLY when BOTH conditions hold:
    #   1. We don't have enough confident context  (needs_human=True)
    #   2. The intent is genuinely sensitive
    # Plain refund/policy questions that ARE covered in the KB are answered normally.
    is_sensitive = intent in SENSITIVE_INTENTS
    should_escalate = is_sensitive

    # Also escalate when we found absolutely nothing (no docs at all)
    if not docs and intent != "general":
        should_escalate = True

    if should_escalate:
        print("  → Escalating to human review.")
        return {
            "draft_answer": (
                "I want to make sure you get the most accurate answer. "
                "I'm escalating your query to a human support agent who will follow up shortly."
            ),
            "escalated": True,
        }

    # ── RAG generation ────────────────────────────────────────────────────────
    if docs:
        context_text = "\n\n---\n\n".join(doc.page_content for doc in docs)
    else:
        context_text = "No specific information found in the knowledge base."

    prompt = (
        "You are a friendly, helpful customer support assistant.\n"
        "Answer the user's question using ONLY the context below.\n"
        "Be concise, clear, and conversational. "
        "If the context doesn't contain the answer, say so politely.\n\n"
        f"Context:\n{context_text}\n\n"
        f"User question: {query}\n\n"
        "Answer:"
    )
    answer = _get_llm().invoke(prompt).content.strip()
    print(f"  draft (first 120 chars): {answer[:120]}")
    return {"draft_answer": answer, "escalated": False}


# ── Node: human_review ────────────────────────────────────────────────────────
def human_review_node(state: RAGState) -> dict:
    """
    Interrupt point.  LangGraph pauses HERE and waits for the admin to call
    graph.update_state(..., as_node="human_review") before resuming.
    """
    print("--- NODE: human_review (paused) ---")
    return {}


# ── Node: finalize ────────────────────────────────────────────────────────────
def finalize_node(state: RAGState) -> dict:
    print("--- NODE: finalize ---")
    answer = state.get("draft_answer") or "I'm sorry, something went wrong. Please try again."
    return {"final_answer": answer}


# ── Routing function ──────────────────────────────────────────────────────────
def route_after_generate(state: RAGState) -> Literal["human_review", "finalize"]:
    if state.get("escalated", False):
        print("--- ROUTE → human_review ---")
        return "human_review"
    print("--- ROUTE → finalize ---")
    return "finalize"


# ── Graph factory ─────────────────────────────────────────────────────────────
def create_graph():
    wf = StateGraph(RAGState)

    wf.add_node("retrieve",       retrieve_node)
    wf.add_node("classify_intent", classify_intent_node)
    wf.add_node("generate",       generate_node)
    wf.add_node("human_review",   human_review_node)
    wf.add_node("finalize",       finalize_node)

    wf.add_edge(START,             "retrieve")
    wf.add_edge("retrieve",        "classify_intent")
    wf.add_edge("classify_intent", "generate")
    wf.add_conditional_edges(
        "generate",
        route_after_generate,
        {"human_review": "human_review", "finalize": "finalize"},
    )
    wf.add_edge("human_review", "finalize")
    wf.add_edge("finalize",     END)

    memory = MemorySaver()
    return wf.compile(checkpointer=memory, interrupt_before=["human_review"])


if __name__ == "__main__":
    g = create_graph()
    print("✅ Graph compiled successfully.")