# HW3.py — URL-aware streaming chatbot with pluggable memory + multi-vendor LLMs
# Providers: OpenAI, Mistral (small latest), Gemini (2.0-flash-lite)

import streamlit as st
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import tiktoken

# Optional providers
try:
    from mistralai import Mistral
except Exception:
    Mistral = None

try:
    import google.generativeai as genai
except Exception:
    genai = None


# ---------- Page ----------
st.set_page_config(page_title="HW 3 — Multi-Vendor Chatbot", page_icon="🧠")
st.title("🧠 HW 3 — Multi-Vendor Chatbot (URLs + memory + streaming)")

# ---------- Secrets ----------
def get_secret(name: str) -> str:
    try:
        return st.secrets[name].strip().replace("\r", "").replace("\n", "")
    except KeyError:
        return ""

OPENAI_KEY  = get_secret("OPENAI_API_KEY")
MISTRAL_KEY = get_secret("MISTRAL_API_KEY")
GEMINI_KEY  = get_secret("GEMINI_API_KEY")

# ---------- Sidebar: inputs & options ----------
st.sidebar.header("Sources (URLs)")
url1 = st.sidebar.text_input("URL 1", placeholder="https://example.com/article-1")
url2 = st.sidebar.text_input("URL 2", placeholder="https://example.com/article-2")

st.sidebar.header("Model")
provider = st.sidebar.selectbox(
    "LLM Provider",
    ["OpenAI", "Mistral", "Gemini"],  # three vendors
    index=0,
)
use_flagship = st.sidebar.checkbox("Use flagship model", value=True)

# Model maps
OPENAI_MODELS  = {"flagship": "gpt-4o",               "cheap": "gpt-4o-mini"}
# You asked to use Mistral *small latest*. We’ll use that regardless of tier so it’s always the pick.
MISTRAL_MODELS = {"flagship": "mistral-small-latest", "cheap": "mistral-small-latest"}
# Gemini flagship set to 2.0 flash lite as requested
GEMINI_MODELS  = {"flagship": "gemini-2.0-flash-lite","cheap": "gemini-1.5-flash"}

st.sidebar.header("Conversation Memory")
memory_mode = st.sidebar.selectbox(
    "Memory strategy",
    ["Buffer: last 6 messages", "Conversation summary", "Token buffer: 2000 tokens"],
    index=2,
)

# ---------- Session state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role": "user"|"assistant", "content": "..."}]
if "last_handled_prompt" not in st.session_state:
    st.session_state.last_handled_prompt = None
if "summary" not in st.session_state:
    st.session_state.summary = ""  # running conversation summary (for "Conversation summary" mode)

# ---------- Controls ----------
col1, col2 = st.columns(2)
with col1:
    st.caption("Tip: answers are streamed; switch models & memory to compare.")
with col2:
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.session_state.last_handled_prompt = None
        st.session_state.summary = ""
        st.rerun()

# ---------- Helpers ----------
SYSTEM_PROMPT = (
    "You are a friendly chatbot explaining things so a 10-year-old can understand. "
    "Use short, clear sentences and simple words. "
    "Whenever you answer, finish with exactly this question on a new line: DO YOU WANT MORE INFO? "
    "If the user says yes then provide more information and re-ask DO YOU WANT MORE INFO. "
    "If the user says no ask the user what question can the bot help with. "
    "If you are unsure re-ask DO YOU WANT MORE INFO."
)

def read_url(url: str) -> str:
    if not url:
        return ""
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        soup = BeautifulSoup(r.content, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        lines = [ln.strip() for ln in text.splitlines()]
        return "\n".join([ln for ln in lines if ln])
    except Exception as e:
        return f"[Error fetching {url}: {e}]"

def token_trim(messages, max_tokens=2000, model="gpt-4o-mini"):
    # Use tiktoken for an approximate tokenizer; works well with OpenAI models
    enc = tiktoken.encoding_for_model(model) if model in ["gpt-4o", "gpt-4o-mini"] else tiktoken.get_encoding("cl100k_base")
    total = 0
    result = []
    for m in reversed(messages):
        t = len(enc.encode(m["content"]))
        if total + t > max_tokens:
            break
        result.append(m)
        total += t
    return list(reversed(result))

def last_n(messages, n):
    return messages[-n:] if len(messages) > n else messages

def build_context(messages, mode: str, selected_model_for_tokens: str):
    """Return a list of chat messages according to selected memory strategy."""
    if mode.startswith("Buffer: last 6"):
        return last_n(messages, 6)
    if mode.startswith("Conversation summary"):
        # Prepend the running summary (if any) as a system note, then include last few turns
        context = []
        if st.session_state.summary.strip():
            context.append({"role": "system", "content": f"Conversation so far (summary): {st.session_state.summary}"})
        context.extend(last_n(messages, 6))
        return context
    # Token buffer ~2000 tokens
    return token_trim(messages, max_tokens=2000, model=selected_model_for_tokens)

# Keep the running summary up to date (cheap model, brief)
def update_summary():
    if not st.session_state.messages:
        return
    recent = st.session_state.messages[-10:]
    text_blob = "\n".join([f"{m['role']}: {m['content']}" for m in recent])
    try:
        if OPENAI_KEY:
            oclient = OpenAI(api_key=OPENAI_KEY, timeout=30, max_retries=1)
            resp = oclient.chat.completions.create(
                model=OPENAI_MODELS["cheap"],
                messages=[
                    {"role": "system", "content": "Summarize this chat briefly (<=120 words) capturing key facts & user goals."},
                    {"role": "user", "content": text_blob},
                ],
                timeout=30,
            )
            st.session_state.summary = resp.choices[0].message.content.strip()
    except Exception:
        pass  # best-effort only

def pick_model_name():
    tier = "flagship" if use_flagship else "cheap"
    if provider == "OpenAI":
        return OPENAI_MODELS[tier]
    if provider == "Mistral":
        return MISTRAL_MODELS[tier]  # both map to mistral-small-latest per your request
    if provider == "Gemini":
        return GEMINI_MODELS[tier]
    return OPENAI_MODELS["cheap"]

# ---------- Display history ----------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

# ---------- Chat input ----------
user_prompt = st.chat_input("Ask your question… (the bot will use the URLs + chat memory)")

if user_prompt and user_prompt != st.session_state.last_handled_prompt:
    st.session_state.last_handled_prompt = user_prompt

    # Save + show user message
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)

    # Fetch URL contents (if provided)
    url_texts = []
    text1 = read_url(url1)
    if text1:
        url_texts.append(f"URL1 ({url1}):\n{text1}")
    text2 = read_url(url2)
    if text2:
        url_texts.append(f"URL2 ({url2}):\n{text2}")
    urls_blob = "\n\n".join(url_texts).strip()

    # Build memory context according to selection
    model_for_tokens = "gpt-4o-mini"  # tokenizer baseline for token buffer
    base_context = build_context(st.session_state.messages, memory_mode, model_for_tokens)

    # Compose final messages (system + optional URLs + memory context)
    messages_for_model = [{"role": "system", "content": SYSTEM_PROMPT}]
    if urls_blob:
        # If any provider dislikes system-context blobs, change this role to "user"
        messages_for_model.append(
            {"role": "system", "content": f"Use the following web context when helpful:\n{urls_blob[:25000]}"}  # keep sane size
        )
    messages_for_model.extend(base_context)

    # Choose provider/model
    selected_model = pick_model_name()

    # --------- Stream the response per provider ---------
    with st.chat_message("assistant"):

        def stream_openai():
            if not OPENAI_KEY:
                raise RuntimeError("Missing OPENAI_API_KEY in secrets.")
            client = OpenAI(api_key=OPENAI_KEY, timeout=60, max_retries=2)
            stream = client.chat.completions.create(
                model=selected_model,
                messages=messages_for_model,
                stream=True,
                timeout=60,
            )
            return stream  # Streamlit consumes this via st.write_stream

        def stream_mistral():
            if Mistral is None:
                raise RuntimeError("mistralai SDK not installed.")
            if not MISTRAL_KEY:
                raise RuntimeError("Missing MISTRAL_API_KEY in secrets.")
            mclient = Mistral(api_key=MISTRAL_KEY)
            # mistralai's chat endpoint doesn't expose delta chunks in the same way here;
            # we yield the full text as one chunk (still works with st.write_stream).
            resp = mclient.chat.complete(
                model=selected_model,  # "mistral-small-latest"
                messages=messages_for_model,
                temperature=0.2,
                max_tokens=1200,
            )
            full_text = ""
            try:
                full_text = resp.choices[0].message.content or ""
            except Exception:
                pass
            def gen():
                # Yield once (single-chunk 'stream')
                yield full_text
            return gen()

        def stream_gemini():
            if genai is None:
                raise RuntimeError("google-generativeai not installed.")
            if not GEMINI_KEY:
                raise RuntimeError("Missing GEMINI_API_KEY in secrets.")
            genai.configure(api_key=GEMINI_KEY)
            model = genai.GenerativeModel(selected_model)  # "gemini-2.0-flash-lite" or "gemini-1.5-flash"
            # Flatten chat to a single prompt for simplicity
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
                stream_obj = stream_openai()
                assistant_text = st.write_stream(stream_obj)
            elif provider == "Mistral":
                assistant_text = st.write_stream(stream_mistral())
            else:
                assistant_text = st.write_stream(stream_gemini())
        except Exception as e:
            st.error(f"Generation failed: {e}")
            assistant_text = "(error during generation)"

    # Save assistant reply
    st.session_state.messages.append({"role": "assistant", "content": assistant_text})

    # Update running summary if that memory mode is active
    if memory_mode.startswith("Conversation summary"):
        update_summary()

    st.rerun()
