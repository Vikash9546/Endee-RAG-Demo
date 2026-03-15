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
from dotenv import load_dotenv
import shutil
import requests

# Load local secrets from .env
load_dotenv()

# ── Page Config ─────────────────────────────────────────
st.set_page_config(page_title="Endee AI Knowledge Base", page_icon="⚡", layout="wide")

# Initialize Chat History for RAG
if "messages" not in st.session_state:
    st.session_state.messages = []

INDEX_NAME = "knowledge_base"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
DIMENSION = 384
SIMILARITY_THRESHOLD = 0.5  # Max allowed distance for a 'relevant' match

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
        ext = os.path.splitext(uploaded.name)[1].lower()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        if ext in [".pdf", ".md", ".txt"]:
            # Text Processing
            text = extract_text(tmp_path, uploaded.name)
            chunks = chunk_text(text)
            vectors = model.encode([c for c in chunks], show_progress_bar=False)
            payloads = []
            for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
                payloads.append({
                    "id": f"text::{uploaded.name}::{i}",
                    "vector": vec.tolist(),
                    "meta": {"text": chunk, "source": uploaded.name, "type": "text"},
                })
            index.upsert(payloads)

        os.unlink(tmp_path)
        progress.progress((fi + 1) / len(uploaded_files), text=f"Processed {uploaded.name}")

    st.sidebar.success(f"✅ Ingested {len(uploaded_files)} file(s) into AI Knowledge Assistant!")

st.sidebar.markdown("---")

st.sidebar.markdown("---")
st.sidebar.title("🛠️ Project Controls")
if st.sidebar.button("🚀 Seed Product Catalog"):
    # Seed Recommendations
    REC_INDEX_NAME = "endee_recommendations_ui"
    try: client.create_index(name=REC_INDEX_NAME, dimension=384, space_type="cosine", precision=Precision.FLOAT32)
    except: pass
    rec_i = client.get_index(name=REC_INDEX_NAME)
    RECS_CAT = [
        {"id": "p1", "desc": "Wireless Noise Cancelling Headphones, over-ear, black", "category": "Electronics"},
        {"id": "p2", "desc": "Running Shoes, lightweight, breathable mesh, blue", "category": "Footwear"},
        {"id": "p3", "desc": "Yoga Mat, non-slip, eco-friendly cork, 5mm thick", "category": "Fitness"},
        {"id": "p4", "desc": "Smartwatch with Heart Rate Monitor and GPS, waterproof", "category": "Electronics"},
        {"id": "p5", "desc": "Organic Green Tea, loose leaf, 100g", "category": "Groceries"},
        {"id": "p6", "desc": "Ergonomic Office Chair with Lumbar Support, adjustable height", "category": "Home Office"},
        {"id": "p7", "desc": "Cast Iron Skillet, pre-seasoned, 12-inch, heavy duty", "category": "Kitchen"},
        {"id": "p8", "desc": "Sci-Fi Novel: 'The Last Frontier', Hardcover first edition", "category": "Books"},
        {"id": "p9", "desc": "Portable Waterproof Speaker with 20-hour battery life", "category": "Outdoor"},
        {"id": "p10", "desc": "Dimmable LED Desk Lamp with Wireless Charging base", "category": "Home Office"},
        {"id": "p11", "desc": "Electric Gooseneck Kettle, temperature control, stainless steel", "category": "Kitchen"},
        {"id": "p12", "desc": "Scented Soy Candle, Lavender and Eucalyptus, 40-hour burn", "category": "Home Decor"}
    ]
    rec_i.upsert([{"id": p["id"], "vector": model.encode([p["desc"]])[0].tolist(), "meta": p} for p in RECS_CAT])
    st.sidebar.success("✅ Recommendations Catalog Synchronized!")

st.sidebar.markdown("---")
st.sidebar.title("🎮 App Navigation")
app_mode = st.sidebar.radio("Select AI Feature:", [
    "🌐 Unified AI Search Dashboard",
    "🤖 AI Knowledge Assistant",
    "🛍️ Product Recommendations",
    "🕵️ Agentic AI Memory"
])
st.sidebar.markdown("---")

# ── Main Area Routing ────────────────────────────────────

if app_mode == "🌐 Unified AI Search Dashboard":
    st.title("🌐 Unified AI Search Dashboard")
    st.markdown("One search powered by **Endee Vector DB**. Retrieves Knowledge and Products simultaneously.")
elif app_mode == "🤖 AI Knowledge Assistant":
    st.title("🤖 AI Knowledge Assistant")
    st.markdown("Ask deep questions about your uploaded documents. Endee retrieves context for accurate LLM answers.")
elif app_mode == "🛍️ Product Recommendations":
    st.title("🛍️ Product Recommendations")
    st.markdown("Semantic intent-based search for products in the catalog.")
elif app_mode == "🕵️ Agentic AI Memory":
    st.title("🕵️ Ghost-Protocol: Agentic AI Memory")
    st.markdown("This mode simulates an **Autonomous Agent** that uses Endee as its Long-Term Memory to handle server incidents.")
    
    AGENT_INDEX = "agentic_incident_memory"
    try: client.create_index(name=AGENT_INDEX, dimension=384, space_type="cosine", precision=Precision.FLOAT32)
    except: pass
    agent_i = client.get_index(name=AGENT_INDEX)

    if st.button("🔧 Seed Agent Memory (Clean Slate)"):
        try: client.delete_index(AGENT_INDEX)
        except: pass
        client.create_index(name=AGENT_INDEX, dimension=384, space_type="cosine", precision=Precision.FLOAT32)
        agent_i = client.get_index(name=AGENT_INDEX)
        
        past_incidents = [
            {"error_str": "Postgres Connection Refused 5432", "solution": "Restarted pg_ctl and increased max_connections.", "difficulty": "Easy"},
            {"error_str": "OOMKilled: Pod memory limit", "solution": "Memory leak detected. Requires senior SRE profile.", "difficulty": "Hard"},
            {"error_str": "AWS S3 Access Denied 403", "solution": "IAM role restored via Terraform.", "difficulty": "Easy"}
        ]
        agent_i.upsert([{"id": f"inc_{i}", "vector": model.encode([p["error_str"]])[0].tolist(), "meta": p} for i, p in enumerate(past_incidents)])
        st.success("Agent Memory Reset & Seeded!")

    incident = st.text_input("🚨 Enter a simulated server error signature:", "Database is crashing. Connection timed out on port 5432")
    
    if st.button("Run Agent Loop"):
        status_box = st.empty()
        status_box.info("🤖 **Agent State**: Analyzing incoming alert signature...")
        time.sleep(1)
        
        status_box.warning("🔍 **Step 1: Consulting Endee Memory...** (Looking for past solutions)")
        query_vec = model.encode([incident])[0].tolist()
        results = agent_i.query(vector=query_vec, top_k=1)
        time.sleep(1.5)

        if results and results[0].get('distance', 1.0) <= 0.45:
            match = results[0].get('meta', {})
            err_name = match.get('error_str', 'Unknown Signature')
            sol_name = match.get('solution', 'No solution steps found')
            diff_level = match.get('difficulty', 'Hard')

            status_box.success(f"✅ **Step 2: Memory Match Found!** Similar issue found: *'{err_name}'*")
            time.sleep(1)
            
            st.markdown("### 🤖 Agent Decision Engine")
            if diff_level == "Easy":
                st.balloons()
                st.success(f"**DECISION: AUTO-FIX 🛠️**\n\nI remember this! Executing known fix: `{sol_name}`")
            else:
                st.warning(f"**DECISION: ESCALATE w/ CONTEXT ⚠️**\n\nI found a match, but the difficulty is '{diff_level}'. Escalating to Human SRE with past context: *{sol_name}*")
        else:
            status_box.error("❌ **Step 2: No Memory Match Found.** This is a novel incident.")
            st.markdown("### 🤖 Agent Decision Engine")
            st.error("**DECISION: EMERGENCY ESCALATE ☎️**\n\nThis error signature is unknown to my internal database. Paging human on-call immediately.")


# Display prior messages if in Knowledge Assistant mode
if app_mode == "🤖 AI Knowledge Assistant":
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# Unified search input
if prompt := st.chat_input(f"Enter your query for {app_mode}..."):
    
    # Store user message for RAG Assistant
    if app_mode == "🤖 AI Knowledge Assistant":
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

    # 📝 1. AI Knowledge Assistant Section (CORE RAG)
    if app_mode in ["🌐 Unified AI Search Dashboard", "🤖 AI Knowledge Assistant"]:
        if app_mode == "🌐 Unified AI Search Dashboard": st.subheader("🤖 AI Knowledge Assistant")
        
        with st.spinner("🧠 RAG Pipeline: Retrieving Context from Endee..."):
            try:
                kb_index = client.get_index(name=INDEX_NAME)
                query_vec = model.encode([prompt])[0].tolist()
                kb_results = kb_index.query(vector=query_vec, top_k=3)
            except: kb_results = []

        if kb_results:
            # Context Extraction & Citation Prep
            contexts = []
            sources = set()
            for m in kb_results:
                meta = m.get("meta", {})
                txt = meta.get("text", "")
                src = meta.get("source", "Unknown")
                contexts.append(f"[Source: {src}] {txt}")
                sources.add(src)
            
            context_block = "\n\n---\n\n".join(contexts)
            
            # Chat memory integration
            chat_history_str = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-5:]])
            
            llm_prompt = f"""
            You are a helpful AI Assistant. Answer the user's question using ONLY the provided context from the Endee Vector Database.
            If the answer isn't in the context, say you don't know based on the documents.
            Always cite the source files at the end of your answer.

            --- CONTEXT ---
            {context_block}

            --- RECENT CHAT HISTORY ---
            {chat_history_str}

            Question: {prompt}
            Answer:
            """
            
            gemini_key = os.environ.get("GEMINI_API_KEY")
            response_text = None
            
            if gemini_key:
                try:
                    from google import genai
                    gen_client = genai.Client(api_key=gemini_key)
                    resp = gen_client.models.generate_content(model="gemini-3-flash-preview", contents=llm_prompt)
                    response_text = resp.text
                except:
                    try:
                        resp = gen_client.models.generate_content(model="gemini-2.0-flash", contents=llm_prompt)
                        response_text = resp.text
                    except: pass
            
            if not response_text:
                response_text = "⚠️ *LLM Quota Exceeded. Falling back to Top Semantic Match:* \n\n" + kb_results[0].get('meta', {}).get('text', '')

            # Outcome Rendering
            if app_mode == "🤖 AI Knowledge Assistant":
                with st.chat_message("assistant"):
                    st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
            else:
                st.info(response_text)
                with st.expander("📚 View Retrieved Chunks (Sources)"):
                    for src in sources: st.caption(f"📍 Reference: {src}")
                    for m in kb_results: st.write(m.get('meta', {}).get('text'))
        else:
            if app_mode == "🤖 AI Knowledge Assistant":
                with st.chat_message("assistant"):
                    st.write("I searched my memory but found no relevant documents to answer that.")
            else:
                st.caption("No knowledge base results found.")

    if app_mode == "🌐 Unified AI Search Dashboard": st.divider()

    # 🛍️ 2. Product Recommendations Section
    if app_mode in ["🌐 Unified AI Search Dashboard", "🛍️ Product Recommendations"]:
        col_rec = st.container()

        with col_rec:
            if app_mode == "🌐 Unified AI Search Dashboard": st.subheader("🛍️ Product Recommendations")
            with st.spinner("Finding similar products..."):
                try:
                    rec_i = client.get_index(name="endee_recommendations_ui")
                    unfiltered_results = rec_i.query(vector=model.encode([prompt])[0].tolist(), top_k=2 if app_mode == "🌐 Unified AI Search Dashboard" else 4)
                    rec_results = [r for r in unfiltered_results if r.get('distance', 1.0) <= SIMILARITY_THRESHOLD]
                    
                    if not rec_results: 
                        st.warning("🔍 I have no such type exist in my database (Product)")
                    else:
                        cols = st.columns(2) if app_mode != "🌐 Unified AI Search Dashboard" else [st.container()]
                        for i, m in enumerate(rec_results):
                            meta = m.get('meta', {})
                            with (cols[i % 2] if app_mode != "🌐 Unified AI Search Dashboard" else st.container()):
                                st.success(f"**{meta.get('category')}**\n\n{meta.get('desc')}")
                except:
                    st.caption("No recommendations found. Use Sidebar to seed catalog.")




