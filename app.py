"""
app.py — Streamlit Chatbot UI for the Endee RAG Knowledge Base
================================================================
Upload PDFs / Markdown / Text → ingest into Endee → ask questions → get AI answers.
"""

import os
import time
import tempfile
import fitz  # PyMuPDF
import streamlit as st
from sentence_transformers import SentenceTransformer
from endee import Endee, Precision

# ── Page Config ─────────────────────────────────────────
st.set_page_config(page_title="Endee AI Knowledge Base", page_icon="⚡", layout="wide")

INDEX_NAME = "knowledge_base"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
DIMENSION = 384

# ── Cached Resources ────────────────────────────────────

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def get_endee():
    return Endee()

model = load_model()
client = get_endee()

# ── Helper Functions ─────────────────────────────────────

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start : start + size])
        start += size - overlap
    return chunks

def extract_text(filepath, filename):
    if filename.lower().endswith(".pdf"):
        doc = fitz.open(filepath)
        text = "".join(page.get_text() for page in doc)
        doc.close()
        return text
    else:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

def ensure_index():
    try:
        return client.get_index(name=INDEX_NAME)
    except Exception:
        try:
            client.create_index(name=INDEX_NAME, dimension=DIMENSION, space_type="cosine", precision=Precision.FLOAT32)
        except Exception:
            pass
        return client.get_index(name=INDEX_NAME)

# ── Sidebar: Document Upload ─────────────────────────────

st.sidebar.title("📁 Upload Documents")
st.sidebar.markdown("Upload **PDFs**, **Markdown**, or **Text** files to build your knowledge base.")

uploaded_files = st.sidebar.file_uploader(
    "Choose files", type=["pdf", "md", "txt"], accept_multiple_files=True
)

if st.sidebar.button("🚀 Ingest into Endee", disabled=not uploaded_files):
    index = ensure_index()
    all_payloads = []

    progress = st.sidebar.progress(0, text="Processing files...")
    for fi, uploaded in enumerate(uploaded_files):
        # Save to temp file for PyMuPDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        text = extract_text(tmp_path, uploaded.name)
        chunks = chunk_text(text)
        vectors = model.encode([c for c in chunks], show_progress_bar=False)

        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            all_payloads.append({
                "id": f"{uploaded.name}::chunk-{i}",
                "vector": vec.tolist(),
                "meta": {"text": chunk, "source": uploaded.name, "chunk_index": i},
            })

        os.unlink(tmp_path)
        progress.progress((fi + 1) / len(uploaded_files), text=f"Processed {uploaded.name}")

    if all_payloads:
        BATCH = 100
        for i in range(0, len(all_payloads), BATCH):
            index.upsert(all_payloads[i : i + BATCH])

        st.sidebar.success(f"✅ Ingested {len(all_payloads)} chunks from {len(uploaded_files)} file(s)!")
    else:
        st.sidebar.warning("No text extracted from the uploaded files.")

st.sidebar.markdown("---")

st.sidebar.markdown("---")
st.sidebar.title("🛠️ Features")
app_mode = st.sidebar.radio("Choose AI Demo:", ["1. Chatbot Knowledge Base", "2. AI Recommendations Engine"])
st.sidebar.markdown("---")
# ── Main Area Routing ────────────────────────────────────

if app_mode == "1. Chatbot Knowledge Base":
    st.title("⚡ Endee AI Knowledge Base")
st.markdown("Ask questions about your uploaded documents. Endee retrieves the most relevant context, and the LLM generates an answer.")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # ── Retrieve from Endee ──────────────────────────
    with st.chat_message("assistant"):
        with st.spinner("Searching Endee for relevant context..."):
            try:
                index = client.get_index(name=INDEX_NAME)
            except Exception:
                st.error("Index not found. Please upload and ingest documents first using the sidebar.")
                st.stop()

            query_vec = model.encode([prompt])[0].tolist()

            start_t = time.time()
            results = index.query(vector=query_vec, top_k=3)
            latency = (time.time() - start_t) * 1000

        if not results:
            response_text = "I couldn't find any relevant context in the knowledge base. Please upload some documents first."
            st.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            st.stop()

        # Show retrieved context
        contexts = []
        with st.expander(f"📎 Retrieved {len(results)} chunks from Endee ({latency:.1f} ms)", expanded=False):
            for i, match in enumerate(results):
                meta = match.get("meta", {})
                text = meta.get("text", "")
                source = meta.get("source", "unknown")
                dist = match.get("distance", 0)
                contexts.append(text)
                st.markdown(f"**[{i+1}] {source}** — distance: `{dist:.4f}`")
                st.text(text[:300])
                st.divider()

        # ── Generate via LLM (OpenAI → Gemini → raw fallback) ──
        context_block = "\n\n---\n\n".join(contexts)
        llm_prompt = (
            f"Use the following context to answer the question: "
            f"{context_block}. "
            f"Question: {prompt}"
        )

        response_text = None
        openai_key = os.environ.get("OPENAI_API_KEY")
        gemini_key = os.environ.get("GEMINI_API_KEY")

        # Try OpenAI
        if openai_key and not response_text:
            with st.spinner("Generating answer with OpenAI GPT-4o-mini..."):
                try:
                    from openai import OpenAI
                    llm = OpenAI(api_key=openai_key)
                    resp = llm.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "You are a helpful technical knowledge assistant. Answer strictly based on the provided context."},
                            {"role": "user", "content": llm_prompt},
                        ],
                        temperature=0.4,
                        max_tokens=600,
                    )
                    response_text = resp.choices[0].message.content
                except Exception as e:
                    st.caption(f"⚠️ OpenAI failed: {e}. Trying Gemini...")

        # Try Gemini (free) — tries multiple models for quota resilience
        if gemini_key and not response_text:
            with st.spinner("Generating answer with Google Gemini..."):
                try:
                    import google.generativeai as genai
                    import time as _time
                    genai.configure(api_key=gemini_key)
                    last_err = None
                    for mname in ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"]:
                        try:
                            gmodel = genai.GenerativeModel(mname)
                            resp = gmodel.generate_content(llm_prompt)
                            response_text = resp.text
                            break
                        except Exception as e:
                            last_err = e
                            _time.sleep(2)
                            continue
                    
                    if not response_text and last_err:
                        st.caption(f"⚠️ Gemini Quota Exceeded or Failed: {last_err}")
                except Exception as e:
                    st.caption(f"⚠️ Gemini Setup failed: {e}")

        # Fallback: raw context
        if not response_text:
            response_text = (
                "⚠️ *LLM Quota Exceeded (or No API key). Falling back to pure Endee Vector Search results:*\n\n"
                "**Raw retrieved context from Endee:**\n\n"
                + "\n\n---\n\n".join(contexts)
            )

        st.markdown(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})

elif app_mode == "2. AI Recommendations Engine":
    st.title("🛍️ AI Recommendations Engine")
    st.markdown("Endee searches for similar products based on a natural language user profile description.")
    
    REC_INDEX_NAME = "endee_recommendations_ui"
    
    try:
        rec_index = client.get_index(name=REC_INDEX_NAME)
    except Exception:
        client.create_index(name=REC_INDEX_NAME, dimension=384, space_type="cosine", precision=Precision.FLOAT32)
        rec_index = client.get_index(name=REC_INDEX_NAME)
        
    PRODUCTS = [
        {"id": "p1", "desc": "Wireless Noise Cancelling Headphones, over-ear, black", "category": "Electronics"},
        {"id": "p2", "desc": "Running Shoes, lightweight, breathable mesh, blue", "category": "Footwear"},
        {"id": "p3", "desc": "Yoga Mat, non-slip, eco-friendly cork, 5mm thick", "category": "Fitness"},
        {"id": "p4", "desc": "Smartwatch with Heart Rate Monitor and GPS, waterproof", "category": "Electronics"},
        {"id": "p5", "desc": "Organic Green Tea, loose leaf, 100g", "category": "Groceries"}
    ]
    
    with st.spinner("Ensuring product catalog is loaded into Endee..."):
        payloads = [{"id": p["id"], "vector": model.encode([p["desc"]])[0].tolist(), "meta": p} for p in PRODUCTS]
        rec_index.upsert(payloads)
        
    st.markdown("##### Current Vector Product Catalog:")
    st.table(PRODUCTS)
    
    user_interest = st.text_input("Describe your interests (Intent-based Semantic Search):", "I want to track my heart rate while exercising outside.")
    
    if st.button("Get Recommendations"):
        with st.spinner("Finding closest vectors in Endee..."):
            user_vec = model.encode([user_interest])[0].tolist()
            results = rec_index.query(vector=user_vec, top_k=2)
            
            st.markdown("### 🎯 Top Recommendations from Endee:")
            col1, col2 = st.columns(2)
            
            for i, match in enumerate(results):
                meta = match.get('meta', {})
                dist = match.get('distance', 0)
                
                with (col1 if i % 2 == 0 else col2):
                    st.info(f"**{meta.get('category', '')}**\n\n{meta.get('desc', '')}\n\n*Similarity Score: {dist:.4f}*")
