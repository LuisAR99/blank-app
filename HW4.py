import streamlit as st
from openai import OpenAI
import tiktoken
import uuid
from pathlib import Path
import glob
import os
import sys
from typing import List, Tuple

__import__("pysqlite3")                
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from PyPDF2 import PdfReader

st.set_page_config(page_title="Lab 4b — Course RAG Chatbot", page_icon="🎓")
st.title("🎓 Lab 4b — Course Information Chatbot (RAG)")

PDF_DIR = "docs"                      
CHROMA_PATH = "./ChromaDB_for_lab"    

if "openai_client" not in st.session_state:
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except KeyError:
        st.error("Missing `OPENAI_API_KEY` in Streamlit secrets.")
        st.stop()
    st.session_state.openai_client = OpenAI(api_key=api_key, timeout=60, max_retries=2)
client = st.session_state.openai_client

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_handled_prompt" not in st.session_state:
    st.session_state.last_handled_prompt = None

col1, col2 = st.columns(2)
with col1:
    model_name = st.selectbox("Model", ["gpt-5-mini", "gpt-4o-mini", "gpt-4o"], index=0)
with col2:
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.session_state.last_handled_prompt = None
        # st.session_state.pop("Lab4_vectorDB", None)  # uncomment to rebuild vectors
        st.rerun()

def _get_encoding(model: str):
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")

def trim_messages_to_tokens(messages, max_tokens=200, model="gpt-4o-mini"):
    enc = _get_encoding(model)
    result, total = [], 0
    for m in reversed(messages):
        t = len(enc.encode(m.get("content", "")))
        if total + t > max_tokens:
            break
        result.append(m)
        total += t
    return list(reversed(result))

def _pdf_to_text_from_path(pdf_path: Path) -> str:
    with pdf_path.open("rb") as f:
        reader = PdfReader(f)
        pages = []
        for p in reader.pages:
            try:
                pages.append(p.extract_text() or "")
            except Exception:
                pages.append("")
        return "\n".join(pages).strip()

def _chunk_text(text: str, chunk_chars: int = 2000, overlap: int = 200) -> List[str]:
    text = text.replace("\r", "")
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + chunk_chars, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(end - overlap, 0)
    return chunks

def _find_local_pdfs(root_dir: str) -> List[Path]:
    pattern = str(Path(root_dir) / "**" / "*.pdf")
    return [Path(p) for p in glob.glob(pattern, recursive=True)]

def get_or_create_lab4_vdb_from_local(pdf_dir: str):
    if "Lab4_vectorDB" in st.session_state and st.session_state.Lab4_vectorDB is not None:
        return st.session_state.Lab4_vectorDB

    Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

    embedder = OpenAIEmbeddingFunction(
        api_key=client.api_key,
        model_name="text-embedding-3-small"
    )

    collection = chroma_client.get_or_create_collection(
        name="Lab4Collection",
        embedding_function=embedder,
        metadata={"hnsw:space": "cosine"}
    )

    pdf_paths = _find_local_pdfs(pdf_dir)
    if not pdf_paths:
        st.info(f"No PDFs found under `{pdf_dir}`.")
        st.session_state.Lab4_vectorDB = collection
        return collection

    # Add only new docs
    existing_ids = set()
    try:
        existing = collection.get()
        existing_ids = set(existing.get("ids", []))
    except Exception:
        pass

    for i, p in enumerate(pdf_paths):
        doc_key = f"{i:04d}_{p.name}"
        if any(eid.startswith(doc_key) for eid in existing_ids):
            continue

        try:
            raw_text = _pdf_to_text_from_path(p)
        except Exception as e:
            st.warning(f"Could not read {p.name}: {e}")
            continue
        if not raw_text:
            continue

        chunks = _chunk_text(raw_text, chunk_chars=2000, overlap=200)
        ids, docs, metas = [], [], []
        for idx, chunk in enumerate(chunks):
            uid = f"{doc_key}_chunk_{idx}_{uuid.uuid4().hex[:8]}"
            ids.append(uid)
            docs.append(chunk)
            metas.append({
                "doc_key": doc_key,
                "source_path": str(p),
                "source_name": p.name,
                "chunk_index": idx,
                "total_chunks": len(chunks),
            })
        if ids:
            collection.add(ids=ids, documents=docs, metadatas=metas)

    st.session_state.Lab4_vectorDB = collection
    return collection


def retrieve_context(query: str, k_chunks: int = 6) -> Tuple[str, List[str]]:
    """
    Returns (context_text, doc_keys_used).
    Collapses by doc_key to keep a mix if many chunks from same file appear.
    """
    vdb = st.session_state.get("Lab4_vectorDB")
    if not vdb:
        return "", []

    res = vdb.query(
        query_texts=[query],
        n_results=max(k_chunks * 3, k_chunks),   # fetch more then collapse
        include=["documents", "metadatas", "distances"]
    )
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]

    # keep order; allow multiple chunks per doc, but cap total k_chunks
    context_chunks = []
    used_docs = []
    for text, m in zip(docs, metas):
        if not text:
            continue
        context_chunks.append(text.strip())
        used_docs.append(m.get("doc_key", ""))
        if len(context_chunks) >= k_chunks:
            break

    # join with clear separators
    context_text = "\n\n---\n\n".join(context_chunks)
    # unique doc_keys in order
    ordered_doc_keys = []
    seen = set()
    for dk in used_docs:
        if dk and dk not in seen:
            seen.add(dk)
            ordered_doc_keys.append(dk)
    return context_text, ordered_doc_keys


with st.sidebar:
    st.subheader("📚 Course Materials")
    st.caption(f"Indexing PDFs from: `{PDF_DIR}`")
    if "Lab4_vectorDB" not in st.session_state:
        with st.spinner("Building vector DB from local PDFs (persisted)…"):
            vdb = get_or_create_lab4_vdb_from_local(PDF_DIR)
            st.success(f"Vector DB ready. Chunks: ~{vdb.count()}")

# System prompt (RAG-aware)

SYSTEM_PROMPT = (
    "You are a helpful course information assistant for our class. "
    "When a question arrives, if CONTEXT is provided, use it to answer first. "
    "Be concise and student-friendly. "
    "Always disclose whether you used the course materials: "
    "- If context is present, begin your first line with: 'Using course materials:' "
    "- If no context is present, begin with: 'No matching course materials; general answer:' "
    "If the question cannot be answered, say so and suggest what to ask or where to look."
)


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

user_q = st.chat_input("Ask about the course (syllabus, policies, deadlines, etc.)…")

if user_q and user_q != st.session_state.last_handled_prompt:
    st.session_state.last_handled_prompt = user_q
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.write(user_q)

    # RAG retrieve
    context_text, doc_keys = retrieve_context(user_q, k_chunks=6)

    # Build messages: include context as a dedicated system message
    limited_history = trim_messages_to_tokens(
        st.session_state.messages, max_tokens=200, model=model_name
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if context_text:
        messages.append({
            "role": "system",
            "content": f"CONTEXT (course materials):\n{context_text}"
        })
    messages += limited_history

    # Stream answer
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=True,
            timeout=60,
        )
        assistant_text = st.write_stream(stream)

        if doc_keys:
            st.caption("Sources (doc_key order): " + ", ".join(doc_keys))

    st.session_state.messages.append({"role": "assistant", "content": assistant_text})
    st.rerun()
