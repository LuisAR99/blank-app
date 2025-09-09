import streamlit as st
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
try:
    from groq import Groq
except Exception:
    Groq = None

st.set_page_config(page_title="HW 2 — URL Summarizer", page_icon="🧪")
st.title("🧪 HW 2 — URL Summarizer (OpenAI + Groq + Hugging Face)")

# ---------------------------
# Sidebar: Summary Options
# ---------------------------
st.sidebar.header("Summary Options")

summary_choice = st.sidebar.radio(
    "Choose a summary style:",
    ["100-word summary", "Two connected paragraphs", "Five bullet points"],
    index=0,
)

language = st.sidebar.selectbox(
    "Output language:",
    ["English", "Spanish", "Italian"],
    index=0,
)

provider = st.sidebar.selectbox(
    "LLM Provider:",
    ["OpenAI", "Groq", "Hugging Face"],
    index=0,
)

use_advanced = st.sidebar.checkbox("Use Advanced Model", value=False)

# Model maps
OPENAI_MODELS = {"advanced": "gpt-4o", "basic": "gpt-4o-mini"}
GROQ_MODELS   = {"advanced": "llama-3.1-70b-versatile", "basic": "llama-3.1-8b-instant"}
HF_MODELS     = {"advanced": "google/gemma-2-9b-it", "basic": "HuggingFaceH4/zephyr-7b-beta"}

# ---------------------------
# URL input
# ---------------------------
url = st.text_input("Enter a URL to summarize (http/https):", placeholder="https://example.com/article")

# ---------------------------
# Helper functions
# ---------------------------
def read_url_content(target_url: str) -> str:
    """Fetch and clean text from a URL."""
    try:
        resp = requests.get(target_url, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        lines = [ln.strip() for ln in text.splitlines()]
        return "\n".join([ln for ln in lines if ln])
    except requests.RequestException as e:
        st.error(f"Error reading {target_url}: {e}")
        return ""

def build_instruction(choice: str, lang: str) -> str:
    if choice == "100-word summary":
        style = "Summarize the document in about 100 words (≤120), one cohesive paragraph."
    elif choice == "Two connected paragraphs":
        style = "Summarize the document in two connected paragraphs (≈150–250 words total)."
    else:
        style = "Summarize the document in exactly five concise bullet points (one sentence each)."
    return f"{style} Write the output in {lang}. Only return the summary."

# ---------------------------
# Provider implementations
# ---------------------------
def summarize_with_openai(text: str, instruction: str) -> str:
    api_key = st.secrets["OPENAI_API_KEY"]
    client = OpenAI(api_key=api_key, timeout=60, max_retries=2)
    model = OPENAI_MODELS["advanced" if use_advanced else "basic"]
    messages = [
        {"role": "system", "content": "You are a precise summarization assistant."},
        {"role": "user", "content": f"{instruction}\n\n--- DOCUMENT START ---\n{text}\n--- DOCUMENT END ---"},
    ]
    resp = client.chat.completions.create(model=model, messages=messages, timeout=60)
    return resp.choices[0].message.content.strip()

def summarize_with_groq(text: str, instruction: str) -> str:
    if Groq is None:
        raise RuntimeError("Groq SDK not installed. Add `groq` to requirements.")
    api_key = st.secrets["GROQ_API_KEY"]
    client = Groq(api_key=api_key)
    model = GROQ_MODELS["advanced" if use_advanced else "basic"]
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a precise summarization assistant."},
            {"role": "user", "content": f"{instruction}\n\n--- DOCUMENT START ---\n{text}\n--- DOCUMENT END ---"},
        ],
        temperature=0.2,
        max_tokens=1200,
    )
    return resp.choices[0].message.content.strip()

def summarize_with_hf(text: str, instruction: str) -> str:
    api_key = st.secrets["HF_API_KEY"]
    model_id = HF_MODELS["advanced" if use_advanced else "basic"]
    prompt = f"{instruction}\n\n--- DOCUMENT START ---\n{text}\n--- DOCUMENT END ---"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 512, "temperature": 0.2},
        "options": {"wait_for_model": True},
    }
    r = requests.post(f"https://api-inference.huggingface.co/models/{model_id}",
                      headers=headers, json=payload, timeout=90)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, list) and data and "generated_text" in data[0]:
        return data[0]["generated_text"].strip()
    return str(data).strip()

PROVIDER_FN = {
    "OpenAI": summarize_with_openai,
    "Groq": summarize_with_groq,
    "Hugging Face": summarize_with_hf,
}

# ---------------------------
# Main flow
# ---------------------------
if url:
    with st.spinner("Fetching URL…"):
        document = read_url_content(url)

    if not document.strip():
        st.error("No text extracted from the URL.")
        st.stop()

    instruction = build_instruction(summary_choice, language)
    doc_for_prompt = document[:25_000]  # safety cutoff

    try:
        summary = PROVIDER_FN[provider](doc_for_prompt, instruction)
        st.subheader("Summary")
        if summary_choice == "Five bullet points" and not summary.lstrip().startswith("-"):
            lines = [ln for ln in summary.splitlines() if ln.strip()]
            if len(lines) >= 3:
                st.markdown("\n".join([f"- {ln}" for ln in lines]))
            else:
                st.write(summary)
        else:
            st.write(summary)
        st.caption(f"Provider: {provider} • Model tier: {'advanced' if use_advanced else 'basic'}")
    except KeyError as ke:
        st.error(f"Missing secret: {ke}. Please add it in .streamlit/secrets.toml.")
    except Exception as e:
        st.error(f"Failed to summarize with {provider}: {e}")
else:
    st.info("Enter a URL above to generate a summary.")
