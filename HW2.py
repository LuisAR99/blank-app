import streamlit as st
import requests
from bs4 import BeautifulSoup

from openai import OpenAI
try:
    from groq import Groq
except Exception:
    Groq = None

try:
    import google.generativeai as genai
except Exception:
    genai = None

st.set_page_config(page_title="HW 2 — URL Summarizer", page_icon="🧪")
st.title("🧪 HW 2 — URL Summarizer (OpenAI + Groq + Gemini)")

# ---------- Sidebar ----------
st.sidebar.header("Summary Options")
summary_choice = st.sidebar.radio(
    "Choose a summary style:",
    ["100-word summary", "Two connected paragraphs", "Five bullet points"],
    index=0,
)
language = st.sidebar.selectbox("Output language:", ["English", "Spanish", "Italian"], index=0)
provider = st.sidebar.selectbox("LLM Provider:", ["OpenAI", "Groq (free)", "Gemini (free)"], index=0)
use_advanced = st.sidebar.checkbox("Use Advanced Model", value=False)

# Model maps
OPENAI_MODELS = {"advanced": "gpt-4o", "basic": "gpt-4o-mini"}
GROQ_MODELS   = {"advanced": "llama-3.3-70b-versatile", "basic": "llama-3.1-8b-instant"}  # updated
GEMINI_MODELS = {"advanced": "gemini-1.5-pro", "basic": "gemini-1.5-flash"}

# ---------- URL input ----------
url = st.text_input("Enter a URL to summarize (http/https):", placeholder="https://example.com/article")

# ---------- Helpers ----------
def read_url_content(target_url: str) -> str:
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

# ---------- Providers ----------
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
        raise RuntimeError("groq SDK not installed. Add `groq` to requirements.txt.")
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

def summarize_with_gemini(text: str, instruction: str) -> str:
    if genai is None:
        raise RuntimeError("google-generativeai not installed. Add `google-generativeai` to requirements.txt.")
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
    model_name = GEMINI_MODELS["advanced" if use_advanced else "basic"]
    model = genai.GenerativeModel(model_name)
    prompt = f"{instruction}\n\n--- DOCUMENT START ---\n{text}\n--- DOCUMENT END ---"
    resp = model.generate_content(prompt)
    return (resp.text or "").strip()

PROVIDER_FN = {
    "OpenAI": summarize_with_openai,
    "Groq (free)": summarize_with_groq,
    "Gemini (free)": summarize_with_gemini,
}

# ---------- Main ----------
if url:
    with st.spinner("Fetching URL…"):
        doc_text = read_url_content(url)

    if not doc_text.strip():
        st.error("No text extracted from the URL.")
    else:
        # keep prompts reasonable
        doc_text = doc_text[:25_000]
        instruction = build_instruction(summary_choice, language)

        try:
            summary = PROVIDER_FN[provider](doc_text, instruction)
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
            st.error(f"Missing secret: {ke}. Add it in .streamlit/secrets.toml.")
        except Exception as e:
            st.error(f"Failed to summarize with {provider}: {e}")
else:
    st.info("Enter a URL above to generate a summary.")
