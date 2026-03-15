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

def vision_ocr_pdf(filepath):
    """Uses Gemini Vision to read handwritten notes from a PDF."""
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        return ""
    
    from google import genai
    import PIL.Image
    import io
    
    gen_client = genai.Client(api_key=gemini_key)
    doc = fitz.open(filepath)
    extracted_text = []
    
    for page in doc:
        # Convert PDF page to Image
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2)) # Higher res for handwriting
        img_data = pix.tobytes("png")
        img = PIL.Image.open(io.BytesIO(img_data))
        
        try:
            prompt = "Please extract all text from this handwritten note or scanned document. Return ONLY the extracted text, no labels."
            response = gen_client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=[prompt, img]
            )
            extracted_text.append(response.text)
        except:
            continue
            
    doc.close()
    return "\n\n".join(extracted_text)

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
            # 1. Try standard digital extraction
            text = extract_text(tmp_path, uploaded.name)
            
            # 2. If empty (Handwritten/Scanned), use Visual OCR Fallback
            if not text.strip() and uploaded.name.lower().endswith(".pdf"):
                with st.sidebar.status(f"🔍 '{uploaded.name}' looks handwritten. Running AI Vision OCR...") as status:
                    text = vision_ocr_pdf(tmp_path)
                    if text.strip():
                        status.update(label="✅ Handwriting Extracted!", state="complete")
                    else:
                        status.update(label="❌ Vision OCR failed.", state="error")
            
            if text.strip():
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
            else:
                st.sidebar.warning(f"⚠️ Could not read any text from {uploaded.name} (even with AI Vision).")

        os.unlink(tmp_path)
        progress.progress((fi + 1) / len(uploaded_files), text=f"Processed {uploaded.name}")

    st.sidebar.success(f"✅ Ingested {len(uploaded_files)} file(s) into AI Knowledge Assistant!")

st.sidebar.markdown("---")

st.sidebar.markdown("---")
st.sidebar.title("🎮 App Navigation")
app_mode = st.sidebar.radio("Select AI Feature:", [
    "🤖 AI Knowledge Assistant",
    "🕵️ Agentic AI Memory"
])
st.sidebar.markdown("---")

# ── Main Area Routing ────────────────────────────────────

if app_mode == "🤖 AI Knowledge Assistant":
    st.title("🤖 AI Knowledge Assistant")
    st.markdown("Ask deep questions about your uploaded documents. Endee retrieves context for accurate LLM answers.")
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

    # 📝 AI Knowledge Assistant Section (CORE RAG)
    if app_mode == "🤖 AI Knowledge Assistant":
        with st.spinner("🧠 RAG Pipeline: Retrieving Context from Endee..."):
            try:
                kb_index = client.get_index(name=INDEX_NAME)
                query_vec = model.encode([prompt])[0].tolist()
                kb_results = kb_index.query(vector=query_vec, top_k=3)
            except: kb_results = []

        # Context Extraction & Citation Prep
        contexts = []
        sources = set()
        if kb_results:
            for m in kb_results:
                meta = m.get("meta", {})
                txt = meta.get("text", "")
                src = meta.get("source", "Unknown")
                contexts.append(f"[Source: {src}] {txt}")
                sources.add(src)
        
        context_block = "\n\n---\n\n".join(contexts) if contexts else "No relevant document snippets were found for this specific query."
        
        # Chat memory integration
        chat_history_str = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-5:]])
        
        llm_prompt = f"""
        You are a friendly and professional AI Knowledge Assistant. 
        
        GUIDELINES:
        1. If the user greets you or wants informal conversation, respond warmly.
        2. For factual questions, prioritize the PROVIDED CONTEXT below.
        3. If the answer is in the context, cite the source file names.
        4. If no relevant context is provided, answer using your general knowledge but mention that you couldn't find specific details in the uploaded documents.

        --- PROVIDED CONTEXT FROM UPLOADED DOCUMENTS ---
        {context_block}

        --- RECENT CHAT HISTORY ---
        {chat_history_str}

        User Question: {prompt}
        Assistant Answer:
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
            if kb_results:
                response_text = "⚠️ *LLM Quota Exceeded. Falling back to Top Matched Chunk:* \n\n" + kb_results[0].get('meta', {}).get('text', '')
            else:
                response_text = "I'm sorry, I encountered a quota issue and couldn't find any documents to help with that query."

        # Outcome Rendering
        with st.chat_message("assistant"):
            st.markdown(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})

        if sources:
            with st.expander("📚 View Retrieved Chunks (Sources)"):
                for src in sources: st.caption(f"📍 Reference: {src}")
                for m in kb_results: st.write(m.get('meta', {}).get('text'))





