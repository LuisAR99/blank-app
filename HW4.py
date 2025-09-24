

import streamlit as st
from openai import OpenAI
import tiktoken
import uuid
from pathlib import Path
import glob
import os
import sys
from typing import List, Tuple

# --- SQLite shim required by Chroma on Streamlit Cloud ---
__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# Optional providers
try:
    from mistralai import Mistral
except Exception:
    Mistral = None

try:
    import google.generativeai as genai
except Exception:
    genai = None

from bs4 import BeautifulSoup
import requests

# ------------------------ Page ---------------------------------
st.set_page_config(page_title="HW 4 — iSchool RAG Chatbot", page_icon="🎓")
st.title("🎓 HW 4 — iSchool Student Organizations Chatbot (RAG)")

# ------------------------ Paths --------------------------------
HTML_DIR = "docs"                        # <- your provided HTML files live here
CHROMA_PATH = "./ChromaDB_for_HW4"       # separate path from lab

# ------------------------ Secrets & Clients ---------------------
def _get_secret(name: str) -> str:
    try:
        return st.secrets[name].strip().replace("\r", "").replace("\n", "")
    except KeyError:
        return ""

OPENAI_API_KEY  = _get_secret("OPENAI_API_KEY")
MISTRAL_API_KEY = _get_secret("MISTRAL_API_KEY")  # optional
GEMINI_API_KEY  = _get_secret("GEMINI_API_KEY")   # optional

if not OPENAI_API_KEY:
    st.error("Missing `OPENAI_API_KEY` in Streamlit secrets. (Needed for embeddings and/or OpenAI LLM.)")
    st.stop()

# We keep an OpenAI client around; Mistral/Gemini are created on demand.
openai_client = OpenAI(api_key=OPENAI_API_KEY, timeout=60, max_retries=2)

# ------------------------ Sidebar (LLMs) ------------------------
st.sidebar.header("Model Provider")
provider = st.sidebar.selectbox("Choose LLM", ["OpenAI", "Mistral", "Gemini"], index=0)
use_flagship = st.sidebar.checkbox("Use flagship model", value=True)

OPENAI_MODELS  = {"flagship": "gpt-4o",               "cheap": "gpt-4o-mini"}
MISTRAL_MODELS = {"flagship": "mistral-small-latest", "cheap": "mistral-small-latest"}
GEMINI_MODELS  = {"flagship": "gemini-2.0-flash-lite","cheap": "gemini-1.5-flash"}

def pick_model_name():
    tier = "flagship" if use_flagship else "cheap"
    if provider == "OpenAI":
        return OPENAI_MODELS[tier]
    if provider == "Mistral":
        return MISTRAL_MODELS[tier]
    if provider == "Gemini":
        return GEMINI_MODELS[tier]
    return OPENAI_MODELS["cheap"]

# ------------------------ Conversation Memory ------------------
# We’ll retain the last 5 Q&A pairs (i.e., last 10 messages) for context.
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role": "user"|"assistant", "content": "..."}]
if "last_handled_prompt" not in st.session_state:
    st.session_state.last_handled_prompt = None

col1, col2 = st.columns(2)
with col1:
    st.caption("Ask about iSchool student organizations. Answers are streamed.")
with col2:
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.session_state.last_handled_prompt = None
        st.rerun()

def _get_encoding(model: str):
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")

def trim_last_n_messages(messages, max_turns=10):
    """Return the last `max_turns` messages (turn ~ one message). 
    We want 5 Q&A pairs -> last 10 messages total."""
    return messages[-max_turns:] if len(messages) > max_turns else messages

# ------------------------ HTML Loading & Chunking ---------------
def _html_file_to_text(html_path: Path) -> str:
    """Read an HTML file into clean text using BeautifulSoup."""
    try:
        raw = html_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        raw = html_path.read_text(errors="ignore")  # fallback, platform-dependent
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n".join([ln for ln in lines if ln])

def _find_local_html(root_dir: str) -> List[Path]:
    pats = [
        str(Path(root_dir) / "**" / "*.html"),
        str(Path(root_dir) / "**" / "*.htm"),
    ]
    files = []
    for p in pats:
        files.extend(glob.glob(p, recursive=True))
    return [Path(p) for p in files]

def _two_chunk_split(text: str, overlap_chars: int = 200) -> List[str]:
    """
    Assignment requirement: Create TWO 'mini-documents' (chunks) per source doc.
    Strategy: Split roughly in half with a small overlap to avoid losing meaning 
    across the split boundary (esp. headings that introduce the next part).
    
    Why this method?
    - Simple, deterministic, satisfies the "two chunks per doc" requirement.
    - The overlap (default 200 chars) keeps continuity for semantic search so 
      relevant context that spans the midpoint isn't lost.
    """
    text = (text or "").replace("\r", "")
    n = len(text)
    if n == 0:
        return []
    if n <= 1200:
        # For very short pages, just return as one chunk, duplicated to meet the "two chunks" idea.
        return [text, text]

    mid = n // 2
    # Left chunk
    start1 = 0
    end1 = min(mid + overlap_chars, n)
    # Right chunk
    start2 = max(mid - overlap_chars, 0)
    end2 = n

    c1 = text[start1:end1].strip()
    c2 = text[start2:end2].strip()
    return [c1, c2]

# ------------------------ Vector DB (create once) ---------------
def get_or_create_vector_db(html_dir: str):
    """
    Creates (if missing) or loads a persistent Chroma collection at CHROMA_PATH.
    Uses OpenAI text-embedding-3-small for embeddings.
    Only adds documents/chunks not already present by ID (so it's safe to rerun).
    """
    Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

    embedder = OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name="text-embedding-3-small",
    )

    collection = chroma_client.get_or_create_collection(
        name="HW4_iSchool_Collection",
        embedding_function=embedder,
        metadata={"hnsw:space": "cosine"},
    )

    # Determine which IDs we already have (so we don't add twice)
    existing_ids = set()
    try:
        existing = collection.get()
        existing_ids = set(existing.get("ids", []))
    except Exception:
        pass

    html_paths = _find_local_html(html_dir)
    if not html_paths:
        st.info(f"No HTML files found under `{html_dir}`. Add files to build the vector DB.")
        return collection

    added = 0
    for i, p in enumerate(sorted(html_paths)):
        doc_key = f"{i:04d}_{p.name}"
        # If any chunk id starting with doc_key exists, skip (already indexed)
        if any(eid.startswith(doc_key) for eid in existing_ids):
            continue

        try:
            raw_text = _html_file_to_text(p)
        except Exception as e:
            st.warning(f"Could not read {p.name}: {e}")
            continue
        if not raw_text:
            continue

        # --- EXACTLY TWO CHUNKS per doc (with small overlap) ---
        chunks = _two_chunk_split(raw_text, overlap_chars=200)
        if not chunks:
            continue

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

        collection.add(ids=ids, documents=docs, metadatas=metas)
        added += len(ids)

    if added:
        st.sidebar.success(f"Indexed {added} new chunks into the vector DB.")
    return collection

# Lazily build (only once per deployment unless new files arrive)
with st.sidebar:
    st.subheader("📚 iSchool Sources")
    st.caption(f"Indexing HTML from: `{HTML_DIR}`")
    if "HW4_vectorDB_ready" not in st.session_state:
        with st.spinner("Building / loading vector DB (persisted)…"):
            vdb = get_or_create_vector_db(HTML_DIR)
            # Lightweight count
            try:
                st.caption(f"Vector DB chunks: ~{vdb.count()}")
            except Exception:
                pass
            st.session_state.HW4_vectorDB_ready = True

# ------------------------ Retrieval -----------------------------
def retrieve_context(query: str, k_chunks: int = 6) -> Tuple[str, List[str]]:
    """
    Returns (context_text, doc_keys_used).
    Retrieves top chunks for the query; we simply cap at k_chunks and later show sources.
    """
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        embedder = OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY, model_name="text-embedding-3-small")
        collection = chroma_client.get_or_create_collection(
            name="HW4_iSchool_Collection",
            embedding_function=embedder,
            metadata={"hnsw:space": "cosine"},
        )
    except Exception as e:
        st.warning(f"Vector DB not available: {e}")
        return "", []

    try:
        res = collection.query(
            query_texts=[query],
            n_results=max(k_chunks, 6),
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        st.warning(f"Vector DB query failed: {e}")
        return "", []

    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]

    context_chunks, used_keys = [], []
    for text, m in zip(docs, metas):
        if not text:
            continue
        context_chunks.append(text.strip())
        used_keys.append(m.get("doc_key", ""))
        if len(context_chunks) >= k_chunks:
            break

    context_text = "\n\n---\n\n".join(context_chunks)
    ordered_doc_keys, seen = [], set()
    for k in used_keys:
        if k and k not in seen:
            seen.add(k)
            ordered_doc_keys.append(k)
    return context_text, ordered_doc_keys

# ------------------------ System Prompt -------------------------
SYSTEM_PROMPT = (
    "You are an iSchool assistant focused on student organizations. "
    "If CONTEXT is provided, use it first. Keep answers short, clear, and helpful. "
    "Always disclose context usage on the first line:\n"
    "- If context is present: 'Using iSchool materials:'\n"
    "- If none: 'No matching iSchool materials; general answer:'\n"
    "If the question cannot be answered, say so and suggest what to ask or where to look."
)

# ------------------------ Chat UI -------------------------------
# Render prior messages
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

user_q = st.chat_input("Ask about iSchool student orgs (events, joining, leadership, etc.)…")

if user_q and user_q != st.session_state.last_handled_prompt:
    st.session_state.last_handled_prompt = user_q
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.write(user_q)

    # Retrieve RAG context
    context_text, doc_keys = retrieve_context(user_q, k_chunks=6)

    # Memory: last 5 Q&A pairs (= last 10 messages)
    limited_history = trim_last_n_messages(st.session_state.messages, max_turns=10)

    # Compose messages for model
    messages_for_model = [{"role": "system", "content": SYSTEM_PROMPT}]
    if context_text:
        messages_for_model.append({"role": "system", "content": f"CONTEXT (iSchool):\n{context_text}"})
    messages_for_model.extend(limited_history)

    # Pick model
    selected_model = pick_model_name()

    with st.chat_message("assistant"):

        def stream_openai():
            client = OpenAI(api_key=OPENAI_API_KEY, timeout=60, max_retries=2)
            return client.chat.completions.create(
                model=selected_model,
                messages=messages_for_model,
                stream=True,
                timeout=60,
            )

        def stream_mistral():
            if Mistral is None or not MISTRAL_API_KEY:
                raise RuntimeError("Mistral not available (missing SDK or API key).")
            mclient = Mistral(api_key=MISTRAL_API_KEY)
            # mistralai chat.complete is non-streaming here; yield single chunk
            resp = mclient.chat.complete(
                model=selected_model,  # "mistral-small-latest"
                messages=messages_for_model,
                temperature=0.2,
                max_tokens=1000,
            )
            txt = ""
            try:
                txt = resp.choices[0].message.content or ""
            except Exception:
                pass
            def gen():
                yield txt
            return gen()

        def stream_gemini():
            if genai is None or not GEMINI_API_KEY:
                raise RuntimeError("Gemini not available (missing SDK or API key).")
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel(selected_model)  # "gemini-2.0-flash-lite" / "gemini-1.5-flash"
            # Flatten messages to a single prompt for simplicity
            prompt = ""
            for m in messages_for_model:
                prompt += f"{m['role'].upper()}: {m['content']}\n"
            resp = model.generate_content(prompt, stream=True)
            def gen():
                for chunk in resp:
                    if hasattr(chunk, "text") and chunk.text:
                        yield chunk.text
            return gen()

        try:
            if provider == "OpenAI":
                assistant_text = st.write_stream(stream_openai())
            elif provider == "Mistral":
                assistant_text = st.write_stream(stream_mistral())
            else:
                assistant_text = st.write_stream(stream_gemini())
        except Exception as e:
            st.error(f"Generation failed: {e}")
            assistant_text = "(error during generation)"

        if doc_keys:
            st.caption("Sources (doc_key order): " + ", ".join(doc_keys))

    # Save and rerun
    st.session_state.messages.append({"role": "assistant", "content": assistant_text})
    st.rerun()
