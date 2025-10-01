import streamlit as st
from openai import OpenAI
import tiktoken
from pathlib import Path
import glob
import sys
from typing import List, Tuple

# --- SQLite shim required by Chroma on Streamlit Cloud ---
__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# Optional providers (same as HW4)
try:
    from mistralai import Mistral
except Exception:
    Mistral = None

try:
    import google.generativeai as genai
except Exception:
    genai = None


# ------------------------ Page ---------------------------------
st.set_page_config(page_title="HW 5 — RAG Chatbot (Short-Term Memory)", page_icon="🧠")
st.title("🧠 HW 5 — Short-Term Memory RAG Chatbot (iSchool Clubs)")

# ------------------------ Paths --------------------------------
HTML_DIR   = "docs"                        # same folder as HW4
CHROMA_PATH = "./ChromaDB_for_HW4"         # reuse HW4 DB (do not rebuild unless missing)

# ------------------------ Secrets & Clients ---------------------
def _secret(name: str) -> str:
    try:
        return st.secrets[name].strip().replace("\r", "").replace("\n", "")
    except KeyError:
        return ""

OPENAI_API_KEY  = _secret("OPENAI_API_KEY")
MISTRAL_API_KEY = _secret("MISTRAL_API_KEY")  # optional
GEMINI_API_KEY  = _secret("GEMINI_API_KEY")   # optional

if not OPENAI_API_KEY:
    st.error("Missing `OPENAI_API_KEY` in Streamlit secrets. (Needed for embeddings and/or OpenAI LLM.)")
    st.stop()

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

# ------------------------ Short-Term Memory --------------------
# Keep the last N messages (short-term memory)
MEMORY_TURNS = 10  # ~5 Q&A pairs
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_prompt" not in st.session_state:
    st.session_state.last_prompt = None

col1, col2 = st.columns(2)
with col1:
    st.caption("Ask about iSchool clubs/orgs. Uses short-term memory + RAG.")
with col2:
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.session_state.last_prompt = None
        st.rerun()

def trim_last_n(messages, n=MEMORY_TURNS):
    return messages[-n:] if len(messages) > n else messages

# ------------------------ Vector DB (load or create if missing) ---------------
def _ensure_vector_db():
    """Load existing HW4 Chroma collection. If missing, create from HTML."""
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    embedder = OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name="text-embedding-3-small"
    )
    collection = chroma_client.get_or_create_collection(
        name="HW4_iSchool_Collection",
        embedding_function=embedder,
        metadata={"hnsw:space": "cosine"},
    )
    return collection

with st.sidebar:
    st.subheader("📚 Source Status")
    try:
        vdb = _ensure_vector_db()
        st.caption(f"Vector DB ready. ~{vdb.count()} chunks indexed.")
    except Exception as e:
        st.error(f"Vector DB error: {e}")
        vdb = None

# ------------------------ Retrieval Function (Required) -----------------------
def get_relevant_club_info(query: str, k_chunks: int = 6) -> Tuple[str, List[str]]:
    """
    Core requirement:
    - Takes a user 'query'
    - Returns a text blob of the most relevant chunks (and the list of source doc_keys)
    This is the ONLY function doing vector search; the LLM only sees its output.
    """
    if not vdb or not query.strip():
        return "", []

    try:
        res = vdb.query(
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
    "Use the provided CONTEXT first. Keep answers short, clear, and helpful. "
    "Always disclose context usage on the first line:\n"
    "- If context is present: 'Using iSchool materials:'\n"
    "- If none: 'No matching iSchool materials; general answer:'\n"
    "If the question cannot be answered, say so and suggest what to ask or where to look."
)

# ------------------------ Render chat history -------------------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

# ------------------------ Chat input & generation ---------------
user_q = st.chat_input("Ask about iSchool student orgs (joining, events, roles, etc.)…")

if user_q and user_q != st.session_state.last_prompt:
    st.session_state.last_prompt = user_q
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.write(user_q)

    # 1) REQUIRED: vector search via function
    context_text, doc_keys = get_relevant_club_info(user_q, k_chunks=6)

    # 2) Build short-term memory (last N messages)
    limited_history = trim_last_n(st.session_state.messages, n=MEMORY_TURNS)

    # 3) Compose messages for the LLM (NO function calling here)
    messages_for_model = [{"role": "system", "content": SYSTEM_PROMPT}]
    if context_text:
        messages_for_model.append({"role": "system", "content": f"CONTEXT (iSchool):\n{context_text}"})
    messages_for_model.extend(limited_history)

    # 4) Pick model
    selected_model = pick_model_name()

    # 5) Stream answer per provider
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

    # 6) Save and rerun
    st.session_state.messages.append({"role": "assistant", "content": assistant_text})
    st.rerun()
