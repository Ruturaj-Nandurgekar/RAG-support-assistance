# RAG Customer Support Assistant

A **Retrieval‑Augmented Generation (RAG)‑based customer support assistant** built with **LangChain**, **LangGraph**, and **Streamlit**.  
Users can upload a **PDF knowledge base** (like policies, manuals, or FAQs), and the assistant answers questions by retrieving relevant information from that document.  
If the system is unsure or the query is sensitive, it can **pause and ask a human agent for review** (Human‑in‑the‑Loop, HITL).

---

##  What this project does

This app is designed to feel like a **real customer support chatbot**:

-  **Pdf‑based knowledge base**  
  - You provide a PDF (e.g., user manual, policy document, FAQ) as the source of truth.  
  - The system **splits the PDF into chunks**, turns them into **embeddings**, and stores them in **ChromaDB** (a vector database).  
  - When you ask a question, the assistant **retrieves the most relevant parts** of the PDF instead of “guessing”.

-  **Conversational RAG assistant**  
  - You chat with the assistant in natural language, just like a real support agent.  
  - The assistant replies in a **friendly, conversational tone**, while staying grounded in the PDF.  
  - If it does not have enough information, it says so instead of making things up.

-  **Human‑in‑the‑Loop (HITL) escalation**  
  - Some questions are **sensitive or risky** (like refunds, policies, or complex cases).  
  - The assistant can decide it needs **human review** and **pauses the workflow**.  
  - A “support agent” (you or someone else) can **see the retrieved context**, **edit the answer**, and **approve or reject** it.

-  **Simple analytics & logging**  
  - Each conversation is logged to a CSV file (`support_cases.csv`), including:
    - user query,
    - assistant answer,
    - confidence score,
    - intent (e.g., billing, refund, policy),
    - whether it was escalated.  
  - This helps understand how often the system needs human help and what kind of questions people ask.

-  **Built with modern RAG + LangGraph**  
  - Uses **LangChain** and **LangGraph** to orchestrate the workflow:
    - PDF → ingestion → retrieval → generation → escalation → final answer.  
  - Uses **Streamlit** for a clean, simple UI that you can run locally and demo easily.

---

##  How to start 

### 1. Prerequisites

You need:
- **Python 3.9+**  
- **A Groq API key** (free at [https://console.groq.com](https://console.groq.com))

Optional:
- A **PDF file** (e.g., `policy.pdf`, `manual.pdf`, or your own FAQ).

### 2. Setup

1. Clone this project (or create the folder structure locally):
   ```bash
   git clone <your-repo-url> RAG
   cd RAG
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set your API key in `.env`:
   ```bash
   echo "GROQ_API_KEY=your_key_here" > .env
   ```

5. Put your PDF in the project root:
   - Name it `knowledge_base.pdf`  
   - Or change the path in `ingest.py` if you prefer a different name.

### 3. Run the ingestion (one time, or after changing the PDF)

This step converts your PDF into a vector database:

```bash
python ingest.py
```

If it succeeds, you’ll see:
- ` Loaded X pages...`  
- ` Split into Y chunks.`  
- ` Added chunks to ChromaDB at chroma_db`.

### 4. Run the Streamlit UI

```bash
streamlit run app.py
```

- The app opens in your browser (usually `http://localhost:8501`).  
- You can:
  - Upload a new PDF in the sidebar.  
  - Ask questions in the chat.  
  - See the **retrieved sources** and **confidence score**.  
  - Review **escalated queries** in the HITL admin panel.

---

## 🧠 What this project shows 

This project demonstrates:

-  **RAG fundamentals**:  
  - PDF → chunking → embeddings → retrieval‑based answers instead of raw LLM hallucination.

-  **LangGraph workflow**:  
  - A multi‑step graph that:
    - retrieves relevant information,
    - decides if escalation is needed based on confidence and intent,
    - pauses for human review, and then continues.

-  **Human‑in‑the‑Loop (HITL)**:  
  - The assistant is not fully “auto‑pilot”; it can escalate when unsure, and a human can review and edit answers.

-  **System design thinking**:  
  - Clear separation: ingestion (`ingest.py`), workflow (`graph.py`), and UI (`app.py`).  
  - Logging and metrics even if they are simple (CSV + sidebar charts).

---

## Project structure (simplified)
RAG/
├── venv/                            # Python virtual environment
├── knowledge_base.pdf               # Your PDF knowledge base
├── chroma_db/                       # Vector database (created by ingest.py)
├── .env                             # Your Groq API key
├── ingest.py                        # PDF → chunks → ChromaDB
├── graph.py                         # LangGraph RAG workflow with HITL
├── app.py                           # Streamlit UI for chat + admin panel
├── requirements.txt                 # Dependencies
└── support_cases.csv                # Logged support cases (CSV)