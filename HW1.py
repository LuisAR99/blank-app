import streamlit as st
from openai import OpenAI
import pdfplumber 

st.title("📄 Document question answering — Lab 2")
st.write(
    "Upload a document below and ask a question about it – GPT will answer! "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
)

uploaded_file = None
question = ""
document = ""  

try:
    api_key = st.secrets["OPENAI_API_KEY"]   
except KeyError:
    st.error(
        "Missing `OPENAI_API_KEY` in Streamlit secrets. "
        "Add it to .streamlit/secrets.toml locally, or set it in Streamlit Cloud → Settings → Secrets."
    )
    st.stop()

client = OpenAI(api_key=api_key, timeout=30, max_retries=2)

st.sidebar.header("Summary Options")
summary_choice = st.sidebar.radio(
    "Choose a summary style:",
    [
        "100-word summary",
        "Two connected paragraphs",
        "Five bullet points",
    ],
    index=0,
)
use_advanced = st.sidebar.checkbox("Use Advanced Model (4o)", value=False)
model = "gpt-4o" if use_advanced else "gpt-4o-mini"


uploaded_file = st.file_uploader("Upload a document (.txt or .pdf)", type=("txt", "pdf"))
question = st.text_area(
    "Now ask a question about the document!",
    placeholder="Can you give me a short summary?",
    disabled=not uploaded_file,
)

submitted = st.button("Ask")

if uploaded_file and question and submitted:
    document = ""

    if hasattr(uploaded_file, "seek"):
        uploaded_file.seek(0)

    if uploaded_file.type == "text/plain":
        document = uploaded_file.read().decode("utf-8", errors="ignore")
    elif uploaded_file.type == "application/pdf":
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                document += page.extract_text() or ""
    else:
        st.error("Unsupported file type. Please upload a .txt or .pdf.")
        st.stop()

    if not document.strip():
        st.error("No text extracted. If this is a scanned PDF, you may need OCR.")
        st.stop()

    messages = [
        {"role": "user", "content": f"Here's a document:\n\n{document}\n\n---\n\n{question}"}
    ]

    stream = client.chat.completions.create(
        model="gpt-4.1",
        messages=messages,
        stream=True,
        timeout=30,
    )
    st.write_stream(stream)

