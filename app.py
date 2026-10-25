import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from pypdf.errors import PdfReadError

from document_processing import (
    chunk_pages,
    extract_pdf_pages,
    format_citation,
    upload_fingerprint,
)
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

st.set_page_config(
    page_title="Document QA Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main {
        background-color: #f8fafc;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .pipeline-step {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .pipeline-step h4 {
        margin: 0 0 0.5rem 0;
        font-size: 0.95rem;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 700;
    }
    .pipeline-badge {
        display: inline-block;
        padding: 0.15rem 0.6rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 700;
        margin-right: 0.5rem;
    }
    .badge-pending { background: #f1f5f9; color: #64748b; border: 1px solid #cbd5e1; }
    .badge-active { background: #dbeafe; color: #1d4ed8; border: 1px solid #bfdbfe; }
    .badge-done { background: #dcfce7; color: #166534; border: 1px solid #bbf7d0; }
    .answer-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .source-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 0.75rem;
        font-size: 0.85rem;
        color: #475569;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        animation: fadeIn 0.2s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .user-msg {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        margin-left: 2rem;
    }
    .bot-msg {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        margin-right: 2rem;
    }
    .status-text {
        font-size: 0.85rem;
        color: #64748b;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pipeline_status" not in st.session_state:
    st.session_state.pipeline_status = {
        "ingest": "pending",
        "chunk": "pending",
        "embed": "pending",
        "ready": False,
    }
if "document_name" not in st.session_state:
    st.session_state.document_name = None
if "active_document_hash" not in st.session_state:
    st.session_state.active_document_hash = None
if "indexed_chunk_settings" not in st.session_state:
    st.session_state.indexed_chunk_settings = None
if "embedding_error" not in st.session_state:
    st.session_state.embedding_error = None


def reset_document_state():
    st.session_state.vector_store = None
    st.session_state.chat_history = []
    st.session_state.pipeline_status = {
        "ingest": "pending",
        "chunk": "pending",
        "embed": "pending",
        "ready": False,
    }
    st.session_state.document_name = None
    st.session_state.pages = []
    st.session_state.chunks = []
    st.session_state.indexed_chunk_settings = None
    st.session_state.embedding_error = None


def get_openai_api_key():
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return api_key

    secrets_paths = (
        Path.cwd() / ".streamlit" / "secrets.toml",
        Path.home() / ".streamlit" / "secrets.toml",
    )
    if not any(path.is_file() for path in secrets_paths):
        return None

    try:
        return st.secrets["OPENAI_API_KEY"]
    except KeyError:
        return None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Controls")
    st.markdown("### 1. Upload Document")
    pdf = st.file_uploader("Upload a PDF file", type="pdf", key="pdf_uploader")

    st.markdown("### 2. Pipeline Settings")
    chunk_size = st.slider("Chunk size", 500, 2000, 1000, 100)
    chunk_overlap = st.slider("Chunk overlap", 0, 500, 200, 50)
    k_chunks = st.slider("Top-K retrieved chunks", 1, 10, 3, 1)
    temperature = st.slider("LLM temperature", 0.0, 1.0, 0.0, 0.1)

    st.markdown("---")
    st.markdown("### About")
    st.info(
        "This assistant uses a **RAG pipeline**: "
        "PDF text is extracted, chunked, embedded into a FAISS vector store, "
        "and retrieved to ground answers in your document."
    )
    st.caption("Tech: Streamlit · LangChain · FAISS · OpenAI")

    if st.button("Clear chat history", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("# 📄 Document QA Assistant")
st.markdown("Ask questions about your uploaded PDF and inspect the retrieved excerpts.")
st.markdown("---")

uploaded_hash = (
    upload_fingerprint(pdf.name, pdf.getvalue())
    if pdf is not None
    else None
)
if uploaded_hash != st.session_state.active_document_hash:
    reset_document_state()
    st.session_state.active_document_hash = uploaded_hash

chunk_settings = (chunk_size, chunk_overlap)
if (
    st.session_state.indexed_chunk_settings is not None
    and st.session_state.indexed_chunk_settings != chunk_settings
):
    st.session_state.vector_store = None
    st.session_state.chat_history = []
    st.session_state.chunks = []
    st.session_state.indexed_chunk_settings = None
    st.session_state.embedding_error = None
    st.session_state.pipeline_status = {
        "ingest": "done",
        "chunk": "pending",
        "embed": "pending",
        "ready": False,
    }

# ---------------------------------------------------------------------------
# Pipeline status row
# ---------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

pipeline = [
    ("ingest", "1 · Ingest", "PDF text extraction"),
    ("chunk", "2 · Chunk", "Text splitting"),
    ("embed", "3 · Embed", "Vector storage"),
    ("ready", "4 · Query", "Ask questions"),
]

for i, (key, title, subtitle) in enumerate(pipeline):
    if key == "ready":
        status = "done" if st.session_state.pipeline_status[key] else "pending"
    else:
        status = st.session_state.pipeline_status[key]
    badge_class = f"badge-{status}"
    with [col1, col2, col3, col4][i]:
        st.markdown(
            f"""
            <div class="pipeline-step">
                <span class="pipeline-badge {badge_class}">{status.upper()}</span>
                <h4>{title}</h4>
                <p class="status-text" style="margin:0">{subtitle}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Step 1: Ingest PDF
# ---------------------------------------------------------------------------
if pdf is not None and st.session_state.pipeline_status["ingest"] == "pending":
    with st.spinner("📖 Extracting text from PDF..."):
        try:
            pages = extract_pdf_pages(pdf.getvalue(), pdf.name)
        except (PdfReadError, ValueError) as error:
            st.error(f"Could not process this PDF: {error}")
            st.stop()

    st.session_state.pages = pages
    st.session_state.document_name = pdf.name
    st.session_state.pipeline_status["ingest"] = "done"
    st.rerun()

# Show ingest result
if st.session_state.pipeline_status["ingest"] == "done":
    extracted_characters = sum(len(str(page["text"])) for page in st.session_state.pages)
    st.success(f"✅ Ingested: {st.session_state.document_name} "
               f"({extracted_characters:,} characters across {len(st.session_state.pages)} pages)")

# ---------------------------------------------------------------------------
# Step 2 & 3: Chunk & Embed
# ---------------------------------------------------------------------------
if chunk_overlap >= chunk_size:
    st.error("Chunk overlap must be less than chunk size.")
    st.stop()

if st.session_state.pipeline_status["ingest"] == "done" and st.session_state.pipeline_status["chunk"] == "pending":
    if st.button("🔪 Chunk & Embed Document", type="primary", use_container_width=True):
        st.session_state.pipeline_status["chunk"] = "active"
        st.rerun()

if st.session_state.pipeline_status["chunk"] == "active":
    with st.spinner("Splitting text into chunks..."):
        chunks = chunk_pages(st.session_state.pages, chunk_size, chunk_overlap)
        if not chunks:
            st.error("No text chunks could be created from this PDF.")
            st.stop()
        st.session_state.chunks = chunks
    st.session_state.pipeline_status["chunk"] = "done"
    st.session_state.pipeline_status["embed"] = "active"
    st.rerun()

if st.session_state.pipeline_status["embed"] == "active":
    api_key = get_openai_api_key()
    if not api_key:
        st.session_state.pipeline_status["embed"] = "pending"
        st.session_state.embedding_error = (
            "OpenAI API key is missing. Set OPENAI_API_KEY in your environment, "
            ".env file, or Streamlit secrets."
        )
        st.rerun()
    else:
        with st.spinner("🧠 Creating vector embeddings (FAISS)..."):
            embeddings = OpenAIEmbeddings(api_key=api_key)
            st.session_state.vector_store = FAISS.from_documents(
                st.session_state.chunks, embeddings
            )
            st.session_state.indexed_chunk_settings = chunk_settings
        st.session_state.embedding_error = None
        st.session_state.pipeline_status["embed"] = "done"
        st.session_state.pipeline_status["ready"] = True
        st.rerun()

if st.session_state.pipeline_status["ready"]:
    st.success(
        f"✅ Ready to query — {len(st.session_state.chunks)} chunks indexed in FAISS."
    )

# ---------------------------------------------------------------------------
# Step 4: Query / Chat
# ---------------------------------------------------------------------------
if st.session_state.pipeline_status["ready"]:
    st.markdown("---")
    st.markdown("### 💬 Ask a Question")

    with st.form("question_form", clear_on_submit=True):
        user_question = st.text_input(
            "What do you want to know about this document?",
            placeholder="e.g. What are the termination clauses?",
            key="question_input",
        )
        submitted = st.form_submit_button("Ask")

    if submitted and user_question.strip():
        question = user_question.strip()
        api_key = get_openai_api_key()
        if not api_key:
            st.error(
                "OpenAI API key is missing. Set OPENAI_API_KEY in your environment, "
                ".env file, or Streamlit secrets."
            )
            st.stop()

        with st.spinner("🤔 Retrieving relevant chunks and generating answer..."):
            docs = st.session_state.vector_store.similarity_search(question, k=k_chunks)

            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=temperature,
                api_key=api_key,
            )

            prompt_template = """
            You are a helpful AI assistant that answers questions based ONLY on the provided context.
            If the answer is not in the context, say "I don't know based on the provided document."
            Do not make up answers. Do not invent page numbers or citations.

            Context:
            {context}

            Question: {question}

            Answer:
            """

            context = "\n\n".join(doc.page_content for doc in docs)
            response = llm.invoke(
                prompt_template.format(context=context, question=question)
            ).content

        st.session_state.chat_history.append(("user", question))
        st.session_state.chat_history.append(("bot", response, docs))

    # Render chat history
    for entry in st.session_state.chat_history:
        if entry[0] == "user":
            with st.chat_message("user"):
                st.markdown(entry[1])
        else:
            _, answer, docs = entry
            with st.chat_message("assistant"):
                st.markdown(answer)
            with st.expander("📚 View Source Chunks", expanded=False):
                for i, doc in enumerate(docs):
                    st.markdown(
                        f"**Retrieved excerpt {i + 1} · "
                        f"{format_citation(doc.metadata)}**"
                    )
                    st.text(doc.page_content[:700] + ("..." if len(doc.page_content) > 700 else ""))
else:
    if (
        st.session_state.pipeline_status["ingest"] == "done"
        and st.session_state.pipeline_status["chunk"] == "done"
        and st.session_state.pipeline_status["embed"] == "pending"
    ):
        if st.session_state.embedding_error:
            st.error(st.session_state.embedding_error)
        st.warning("Document chunks are ready, but vector indexing has not completed.")
        if st.button("Retry embedding", type="primary"):
            st.session_state.embedding_error = None
            st.session_state.pipeline_status["embed"] = "active"
            st.rerun()
    elif st.session_state.pipeline_status["ingest"] == "done":
        st.info("Document text is ready. Click **Chunk & Embed Document** to build its search index.")
    else:
        st.markdown(
            """
            <div class="pipeline-step" style="text-align:center">
                <h4 style="color:#64748b">Waiting for document...</h4>
                <p class="status-text">Upload a PDF in the sidebar, then run the pipeline to begin asking questions.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
