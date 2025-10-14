import streamlit as st
import pandas as pd
import numpy as np
import json
import re
import math
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import List, Dict, Any, Tuple

# --- SQLite shim required by Chroma on Streamlit Cloud ---
__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# LLM vendors
from openai import OpenAI
try:
    import google.generativeai as genai
except Exception:
    genai = None

# ------------------------ Page ---------------------------------
st.set_page_config(page_title="HW7 — News Info Bot (RAG)", page_icon="📰", layout="wide")
st.title("📰 HW7 — News Info Bot (RAG for a Global Law Firm)")

# ------------------------ Secrets ------------------------------
def _secret(name: str) -> str:
    try:
        return st.secrets[name].strip().replace("\r","").replace("\n","")
    except KeyError:
        return ""

OPENAI_API_KEY = _secret("OPENAI_API_KEY")
GEMINI_API_KEY = _secret("GEMINI_API_KEY")

if not OPENAI_API_KEY:
    st.error("Missing `OPENAI_API_KEY` in secrets (needed for embeddings and/or OpenAI LLM).")
    st.stop()

# ------------------------ Paths --------------------------------
DATA_PATH   = Path("./news.csv")                # optional default file
CHROMA_PATH = "./ChromaDB_for_HW7"
COLLECTION  = "HW7_News_Collection"

# ------------------------ Sidebar: models & options ------------
st.sidebar.header("Models")
vendor = st.sidebar.selectbox("Vendor", ["OpenAI", "Gemini"], index=0)
tier   = st.sidebar.selectbox("Tier", ["Flagship", "Cheap"], index=0)

OPENAI_MODELS = {"Flagship": "gpt-4o", "Cheap": "gpt-4o-mini"}
GEMINI_MODELS = {"Flagship": "gemini-2.0-flash-lite", "Cheap": "gemini-1.5-flash"}

def pick_model():
    if vendor == "OpenAI":
        return ("openai", OPENAI_MODELS[tier])
    else:
        return ("gemini", GEMINI_MODELS[tier])

prov, model_name = pick_model()
st.sidebar.caption(f"Using: {vendor} → {model_name}")

st.sidebar.header("Query")
mode = st.sidebar.radio(
    "What do you want?",
    ["Most interesting news (for a global law firm)", "News about a topic"],
    index=0
)
topic = st.sidebar.text_input("Topic (only used for topic search)", placeholder="antitrust, sanctions, AI policy…")

max_items = st.sidebar.slider("Max results", 3, 25, 10)

# ------------------------ Utils --------------------------------
def safe_parse_date(s: str) -> datetime | None:
    if not s or not isinstance(s, str): return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(s[:len(fmt)], fmt).replace(tzinfo=timezone.utc)
        except Exception:
            continue
    return None

LEGAL_KEYWORDS = [
    "antitrust","competition","merger","acquisition","M&A","litigation","lawsuit","settlement",
    "regulation","regulatory","SEC","DOJ","FTC","EU Commission","sanction","export control",
    "compliance","FCPA","bribery","privacy","GDPR","CCPA","data protection","intellectual property",
    "patent","trademark","copyright","AI Act","Digital Markets Act","whistleblower","class action",
    "injunction","subpoena","arbitration","tax","ESG","governance"
]

def legal_relevance_heuristic(text: str) -> float:
    if not text: return 0.0
    t = text.lower()
    hits = sum(1 for kw in LEGAL_KEYWORDS if kw.lower() in t)
    return min(1.0, hits / 6.0)

def recency_score(d: datetime | None) -> float:
    if not d: return 0.3
    days = (datetime.now(timezone.utc) - d).days
    return 1.0 / (1.0 + math.exp((days - 30) / 20.0))

# ------------------------ Load CSV -----------------------------
st.subheader("📄 Data")
uploaded = st.file_uploader("Upload a news CSV (or leave blank to use ./news.csv if present)", type=["csv"])
df: pd.DataFrame | None = None

if uploaded is not None:
    df = pd.read_csv(uploaded)
elif DATA_PATH.exists():
    df = pd.read_csv(DATA_PATH)

if df is None:
    st.info("Please upload a CSV with columns like: title, summary, content, url, date, source, tags.")
    st.stop()

# Normalize columns
for col in ["title","summary","content","url","date","source","tags"]:
    if col not in df.columns:
        df[col] = ""

# Parse dates
df["_parsed_date"] = df["date"].apply(safe_parse_date)

# ------------------------ Build/Load Vector DB -----------------
Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

embedder = OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-3-small"
)

collection = chroma_client.get_or_create_collection(
    name=COLLECTION,
    embedding_function=embedder,
    metadata={"hnsw:space":"cosine"},
)

# Add any new rows not yet indexed
existing_ids = set()
try:
    existing = collection.get()
    existing_ids = set(existing.get("ids", []))
except Exception:
    pass

added = 0
ids, docs, metas = [], [], []
for i, row in df.reset_index().iterrows():
    rid = f"row_{i:06d}"
    if rid in existing_ids:
        continue
    text_blob = "\n".join([
        str(row.get("title","")),
        str(row.get("summary","")),
        str(row.get("content","")),
        f"source: {row.get('source','')}",
        f"date: {row.get('date','')}",
        f"tags: {row.get('tags','')}",
    ])
    ids.append(rid)
    docs.append(text_blob)
    metas.append({
        "row_index": int(i),
        "title": str(row.get("title","")),
        "url": str(row.get("url","")),
        "date": str(row.get("date","")),
        "source": str(row.get("source","")),
        "tags": str(row.get("tags","")),
    })

if ids:
    collection.add(ids=ids, documents=docs, metadatas=metas)
    added = len(ids)

st.caption(f"Vector DB ready. Newly indexed: {added}. Total (approx): {collection.count()}.")

# ------------------------ Retrieval ----------------------------
def retrieve_by_topic(query: str, k: int = 30) -> List[Dict[str, Any]]:
    if not query.strip():
        rows = []
        for i, row in df.reset_index().iterrows():
            rows.append({"row_index": int(i), **row.to_dict()})
        rows.sort(key=lambda r: recency_score(r.get("_parsed_date")), reverse=True)
        return rows[:k]
    res = collection.query(query_texts=[query], n_results=k, include=["metadatas","documents","distances"])
    metas = (res.get("metadatas") or [[]])[0]
    out = []
    for m in metas:
        idx = m.get("row_index")
        if idx is None: continue
        r = df.iloc[int(idx)].to_dict()
        r["row_index"] = int(idx)
        out.append(r)
    return out

# ------------------------ LLM Clients --------------------------
openai_client = OpenAI(api_key=OPENAI_API_KEY, timeout=60, max_retries=2)
if genai and GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

def llm_rank_interesting(items: List[Dict[str, Any]], vendor_key: str, model: str) -> List[Dict[str, Any]]:
    """
    Ask the LLM to score 1-10 'interestingness for a global law firm' (JSON only).
    Returns same list with 'llm_score' field added (float).
    """
    pack = []
    for i, r in enumerate(items):
        pack.append({
            "id": i,
            "title": r.get("title","")[:200],
            "summary": r.get("summary","")[:600],
            "source": r.get("source",""),
            "date": r.get("date",""),
            "url": r.get("url",""),
            "tags": r.get("tags",""),
        })
    instr = (
        "You are ranking news for a GLOBAL LAW FIRM. Score each item from 1 (not interesting) to 10 (highly relevant) "
        "considering legal impact (litigation/regulatory/M&A/sanctions/privacy/IP), cross-border scope, and client risk. "
        "Return STRICT JSON as a list of {id, score, rationale} with `score` numeric 1-10. No prose."
    )
    if vendor_key == "openai":
        prompt = [
            {"role":"system","content":instr},
            {"role":"user","content":json.dumps(pack)}
        ]
        resp = openai_client.chat.completions.create(
            model=model,
            messages=prompt,
            temperature=0.2
        )
        txt = resp.choices[0].message.content.strip()
    else:
        gmodel = genai.GenerativeModel(model)
        txt = gmodel.generate_content(f"{instr}\n\n{json.dumps(pack)}").text

    scores = []
    try:
        m = re.search(r"\[.*\]", txt, flags=re.DOTALL)
        if m: scores = json.loads(m.group(0))
        else: scores = json.loads(txt)
    except Exception:
        scores = [{"id": i, "score": 5, "rationale": "fallback"} for i in range(len(items))]

    by_id = {s.get("id"): float(s.get("score",5)) for s in scores if "id" in s}
    for i, r in enumerate(items):
        r["llm_score"] = float(by_id.get(i, 5.0))
    return items

def combine_scores(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Final score = 0.45*LLM + 0.35*recency + 0.20*legal_heuristic  (all 0-1)
    LLM score normalized 1..10 -> 0..1
    """
    out = []
    for r in rows:
        d = r.get("_parsed_date")
        rs = recency_score(d)
        lh = legal_relevance_heuristic(" ".join([
            str(r.get("title","")), str(r.get("summary","")), str(r.get("tags",""))
        ]))
        ls = float(r.get("llm_score", 5.0))
        ls01 = max(0.0, min(1.0, (ls-1)/9.0))
        final = 0.45*ls01 + 0.35*rs + 0.20*lh
        r2 = dict(r)
        r2["_recency"] = rs
        r2["_legal_kw"] = lh
        r2["_final_score"] = final
        out.append(r2)
    out.sort(key=lambda x: x["_final_score"], reverse=True)
    return out

# -------- NEW: Batch summarize top-ranked items (2–3 sentences each) ----------
def llm_summarize_top(items: List[Dict[str, Any]], vendor_key: str, model: str) -> Dict[int, str]:
    """
    Create concise 2–3 sentence summaries for ranked items (global-law-firm lens).
    Returns dict: id -> summary. Uses a single JSON-only call for cost/latency.
    """
    pack = []
    for i, r in enumerate(items):
        # Prefer existing summary; fallback to content (truncate)
        text = str(r.get("summary") or r.get("content") or "")
        text = text[:1200]  # trim for token safety
        pack.append({
            "id": i,
            "title": r.get("title","")[:200],
            "text": text,
            "source": r.get("source",""),
            "date": r.get("date",""),
        })

    instr = (
        "Write a concise 2–3 sentence summary for each article for a GLOBAL LAW FIRM audience. "
        "Emphasize legal/regulatory/litigation/M&A/privacy angles and client risk. "
        "Return STRICT JSON: a list of {id, summary}, where 'summary' ≤ 60 words, no markdown."
    )

    if vendor_key == "openai":
        prompt = [
            {"role":"system","content":instr},
            {"role":"user","content":json.dumps(pack)}
        ]
        resp = openai_client.chat.completions.create(
            model=model,
            messages=prompt,
            temperature=0.2
        )
        txt = resp.choices[0].message.content.strip()
    else:
        gmodel = genai.GenerativeModel(model)
        txt = gmodel.generate_content(f"{instr}\n\n{json.dumps(pack)}").text

    results = {}
    try:
        m = re.search(r"\[.*\]", txt, flags=re.DOTALL)
        arr = json.loads(m.group(0) if m else txt)
        for item in arr:
            iid = int(item.get("id"))
            sm = str(item.get("summary","")).strip()
            if sm:
                results[iid] = sm
    except Exception:
        # Fallback: naive truncation if parsing fails
        for i, r in enumerate(items):
            base = str(r.get("summary") or r.get("content") or "")[:200]
            results[i] = base + ("…" if len(base) == 200 else "")
    return results

# ------------------------ Run Query ----------------------------
st.subheader("🔎 Ask")
colA, colB = st.columns([3,1])
with colA:
    user_q = st.text_input(
        "Your question (e.g., 'find the most interesting news', or 'find news about antitrust')",
        value="find the most interesting news"
    )
with colB:
    run = st.button("Run")

if run:
    with st.spinner("Retrieving and ranking…"):
        if mode.startswith("News about"):
            q = topic.strip() or user_q
            base = retrieve_by_topic(q, k=40)
            if topic.strip():
                t = topic.lower()
                base = [r for r in base if t in (" ".join([
                    str(r.get("title","")), str(r.get("summary","")),
                    str(r.get("content","")), str(r.get("tags",""))
                ]).lower())]
            if not base:
                base = retrieve_by_topic(q, k=40)
        else:
            base = retrieve_by_topic("", k=40)

        seed = base[:20] if len(base) > 20 else base

        # Choose vendor/model for ranking & summaries
        if prov == "gemini" and not GEMINI_API_KEY and vendor == "Gemini":
            st.warning("Gemini API key missing; falling back to OpenAI for ranking/summaries.")
            prov2, model2 = "openai", OPENAI_MODELS["Cheap"]
        else:
            prov2, model2 = prov, model_name

        ranked_input = llm_rank_interesting(seed, prov2, model2)
        combined = combine_scores(ranked_input)

        # ---------- NEW: Generate concise summaries for top-k ----------
        topk = combined[:max_items]
        summaries = llm_summarize_top(topk, prov2, model2)

    # --------------------- Output -------------------------------
    st.subheader("📈 Results")
    for i, r in enumerate(topk, start=1):
        with st.container(border=True):
            st.markdown(f"**{i}. {r.get('title','(no title)')}**")
            meta_line = []
            if r.get('source'): meta_line.append(r['source'])
            if r.get('date'):   meta_line.append(r['date'])
            st.caption(" • ".join(meta_line) if meta_line else " ")
            # NEW: show concise LLM summary
            sm = summaries.get(i-1, "").strip()
            if sm:
                st.write(sm)
            else:
                # fallback to original summary/content if no LLM summary
                if r.get("summary"):
                    st.write(r["summary"])
                elif r.get("content"):
                    st.write(str(r["content"])[:500] + ("…" if len(str(r["content"]))>500 else ""))
            if r.get("url"):
                st.markdown(f"[Link]({r['url']})")
            st.progress(min(1.0, r["_final_score"]))
            with st.expander("Scoring details"):
                st.json({
                    "LLM score (1-10)": r.get("llm_score", 5),
                    "Recency (0-1)": round(r["_recency"],3),
                    "Legal keyword heuristic (0-1)": round(r["_legal_kw"],3),
                    "Final score (0-1)": round(r["_final_score"],3)
                })

    # --------------------- Explanation (stream) -----------------
    st.subheader("🧠 Why these stories?")
    rationale_system = (
        "You are explaining rankings for a global law firm. "
        "Given the top results (titles, sources, dates, and scores), explain succinctly "
        "why these items ranked highly, referencing regulatory/litigation/M&A/privacy relevance and recency. "
        "Keep it under 120 words."
    )
    top_pack = [{"rank": i+1,
                 "title": r.get("title",""),
                 "source": r.get("source",""),
                 "date": r.get("date",""),
                 "score": round(r["_final_score"],3)}
                for i, r in enumerate(topk)]
    if prov2 == "openai":
        stream = openai_client.chat.completions.create(
            model=model2,
            messages=[
                {"role":"system","content": rationale_system},
                {"role":"user","content": json.dumps(top_pack)}
            ],
            stream=True,
            temperature=0.3
        )
        st.write_stream(stream)
    else:
        if genai and GEMINI_API_KEY:
            gmodel = genai.GenerativeModel(model2)
            resp = gmodel.generate_content(
                f"{rationale_system}\n\n{json.dumps(top_pack)}",
                stream=True
            )
            def gen():
                for ch in resp:
                    if hasattr(ch, "text") and ch.text:
                        yield ch.text
            st.write_stream(gen())
        else:
            st.info("Gemini key missing; skipping streamed rationale.")

