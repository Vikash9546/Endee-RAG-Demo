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
import cv2
from PIL import Image
import requests

# Load local secrets from .env
load_dotenv()

# ── Page Config ─────────────────────────────────────────
st.set_page_config(page_title="Endee AI Knowledge Base", page_icon="⚡", layout="wide")

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

@st.cache_resource
def load_clip():
    return SentenceTransformer("clip-ViT-B-32")

model = load_model()
client = get_endee()
clip_m = load_clip()

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

def extract_video_frame(path):
    """Extracts the first frame of a video for vectorization."""
    cap = cv2.VideoCapture(path)
    ret, frame = cap.read()
    cap.release()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)
    return None

# ── Sidebar: Document Upload ─────────────────────────────

st.sidebar.title("📁 Upload Documents")
st.sidebar.markdown("Upload **PDFs**, **Markdown**, or **Text** files to build your knowledge base.")

uploaded_files = st.sidebar.file_uploader(
    "Choose files", type=["pdf", "md", "txt", "png", "jpg", "jpeg", "mp4", "mov"], accept_multiple_files=True
)

if st.sidebar.button("🚀 Ingest into Endee", disabled=not uploaded_files):
    index = ensure_index()
    all_payloads = []

    progress = st.sidebar.progress(0, text="Processing files...")
    
    mm_index_name = "multimodal_kb"
    try: client.create_index(name=mm_index_name, dimension=512, space_type="cosine", precision=Precision.FLOAT32)
    except: pass
    mm_index = client.get_index(name=mm_index_name)

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
        
        elif ext in [".png", ".jpg", ".jpeg"]:
            # Image Processing
            img = Image.open(tmp_path)
            vec = clip_m.encode(img).tolist()
            mm_index.upsert([{
                "id": f"img::{uploaded.name}",
                "vector": vec,
                "meta": {"source": uploaded.name, "type": "image", "url": "local"} # in real app, save path
            }])
            # For demo, we store the file in a local 'uploads' dir to show in UI
            if not os.path.exists("uploads"): os.makedirs("uploads")
            import shutil
            shutil.copy(tmp_path, f"uploads/{uploaded.name}")

        elif ext in [".mp4", ".mov"]:
            # Video Processing
            frame = extract_video_frame(tmp_path)
            if frame:
                vec = clip_m.encode(frame).tolist()
                mm_index.upsert([{
                    "id": f"video::{uploaded.name}",
                    "vector": vec,
                    "meta": {"source": uploaded.name, "type": "video"}
                }])
                if not os.path.exists("uploads"): os.makedirs("uploads")
                shutil.copy(tmp_path, f"uploads/{uploaded.name}")

        os.unlink(tmp_path)
        progress.progress((fi + 1) / len(uploaded_files), text=f"Processed {uploaded.name}")

    st.sidebar.success(f"✅ Ingested {len(uploaded_files)} file(s) into Multi-Modal Intelligence!")

st.sidebar.markdown("---")

st.sidebar.markdown("---")
st.sidebar.title("🛠️ Project Controls")
if st.sidebar.button("🚀 Seed All Catalogs (Recs & Visual)"):
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

    # Seed Multi-Modal
    import requests
    from PIL import Image
    MM_INDEX_NAME = "endee_multimodal_ui"
    try: client.create_index(name=MM_INDEX_NAME, dimension=512, space_type="cosine", precision=Precision.FLOAT32)
    except: pass
    mm_i = client.get_index(name=MM_INDEX_NAME)
    VIS_CAT = [
        {"id": "img_1", "category": "Fashion", "url": "https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?width=300", "desc": "A white plain t-shirt"},
        {"id": "img_2", "category": "Fashion", "url": "https://images.unsplash.com/photo-1503342217505-b0a15ec3261c?width=300", "desc": "A black plain t-shirt"},
        {"id": "img_3", "category": "Footwear", "url": "https://images.unsplash.com/photo-1542291026-7eec264c27ff?width=300", "desc": "Red Nike running shoes"},
        {"id": "img_4", "category": "Pets", "url": "https://images.unsplash.com/photo-1517849845537-4d257902454a?width=300", "desc": "A cute dog looking at camera"},
        {"id": "img_5", "category": "Vehicles", "url": "https://images.unsplash.com/photo-1492144534655-ae79c964c9d7?width=300", "desc": "A sleek silver sports car"}
    ]
    
    @st.cache_resource
    def load_clip():
        return SentenceTransformer("clip-ViT-B-32")
    clip_m = load_clip()

    mm_payloads = []
    for item in VIS_CAT:
        try:
            r = requests.get(item["url"], stream=True, timeout=5)
            img = Image.open(r.raw)
            mm_payloads.append({"id": item["id"], "vector": clip_m.encode(img).tolist(), "meta": item})
        except: pass
    if mm_payloads: mm_i.upsert(mm_payloads)
    st.sidebar.success("✅ Catalogs Synchronized!")

st.sidebar.markdown("---")
st.sidebar.title("🎮 App Navigation")
app_mode = st.sidebar.radio("Select AI Feature:", [
    "🌐 Unified AI Search Dashboard",
    "🤖 AI Knowledge Assistant",
    "🛍️ Product Recommendations",
    "📸 Multi-Media Model",
    "🕵️ Agentic AI Memory"
])
st.sidebar.markdown("---")

# ── Main Area Routing ────────────────────────────────────

if app_mode == "🌐 Unified AI Search Dashboard":
    st.title("🌐 Unified AI Search Dashboard")
    st.markdown("One search powered by **Endee Vector DB**. Retrieves Knowledge, Products, and Visuals simultaneously.")
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


# Unified search input
if prompt := st.chat_input(f"Enter your query for {app_mode}..."):
    
    # 📝 1. AI Knowledge Assistant Section
    if app_mode in ["🌐 Unified AI Search Dashboard", "🤖 AI Knowledge Assistant"]:
        if app_mode == "🌐 Unified AI Search Dashboard": st.subheader("🤖 AI Knowledge Assistant")
        with st.spinner("Searching Knowledge Base..."):
            try:
                kb_index = client.get_index(name=INDEX_NAME)
                query_vec = model.encode([prompt])[0].tolist()
                kb_results = kb_index.query(vector=query_vec, top_k=3)
            except: kb_results = []

        if kb_results:
            contexts = [m.get("meta", {}).get("text", "") for m in kb_results]
            context_block = "\n\n---\n\n".join(contexts)
            
            gemini_key = os.environ.get("GEMINI_API_KEY")
            response_text = None
            llm_prompt = f"Answer based on context: {context_block}\nQuestion: {prompt}"
            if gemini_key:
                try:
                    from google import genai
                    gen_client = genai.Client(api_key=gemini_key)
                    # Use gemini-3-flash-preview as requested by user
                    try:
                        resp = gen_client.models.generate_content(model="gemini-3-flash-preview", contents=llm_prompt)
                        response_text = resp.text
                    except:
                        try:
                            resp = gen_client.models.generate_content(model="gemini-2.0-flash", contents=llm_prompt)
                            response_text = resp.text
                        except: pass
                except: pass
            
            if not response_text:
                response_text = "⚠️ *Quota hit. Showing raw Endee Context:*\n\n" + contexts[0][:500] + "..."
            st.info(response_text)
        else:
            st.caption("No knowledge base results found.")

    if app_mode == "🌐 Unified AI Search Dashboard": st.divider()

    # 🛍️ 2. Product Recommendations Section
    if app_mode in ["🌐 Unified AI Search Dashboard", "🛍️ Product Recommendations"]:
        if app_mode == "🌐 Unified AI Search Dashboard":
            col_rec, col_vis = st.columns(2)
        else:
            col_rec = st.container()

        with col_rec:
            if app_mode == "🌐 Unified AI Search Dashboard": st.subheader("🛍️ Product Recommendations")
            with st.spinner("Finding similar products..."):
                try:
                    rec_i = client.get_index(name="endee_recommendations_ui")
                    unfiltered_results = rec_i.query(vector=model.encode([prompt])[0].tolist(), top_k=2 if app_mode == "🌐 Unified AI Search Dashboard" else 4)
                    
                    # Filter by threshold
                    rec_results = [r for r in unfiltered_results if r.get('distance', 1.0) <= SIMILARITY_THRESHOLD]
                    
                    if not rec_results: 
                        st.warning("🔍 I have no such type exist in my database (Product)")
                    else:
                        # Responsive grid for recommendations
                        cols = st.columns(2) if app_mode != "🌐 Unified AI Search Dashboard" else [st.container()]
                        for i, m in enumerate(rec_results):
                            meta = m.get('meta', {})
                            with (cols[i % 2] if app_mode != "🌐 Unified AI Search Dashboard" else st.container()):
                                st.success(f"**{meta.get('category')}**\n\n{meta.get('desc')}")
                except:
                    st.caption("No recommendations found. Use Sidebar to seed catalog.")

    # 📸 3. Multi-Media Model Section
    if app_mode in ["🌐 Unified AI Search Dashboard", "📸 Multi-Media Model"]:
        if app_mode == "🌐 Unified AI Search Dashboard":
            # We already have col_vis from the st.columns(2) above
            pass
        else:
            col_vis = st.container()

        with col_vis:
            if app_mode == "🌐 Unified AI Search Dashboard": st.subheader("📸 Visual Search")
            with st.spinner("Fetching visual matches..."):
                try:
                    q_vec = clip_m.encode([prompt])[0].tolist()
                    
                    # 1. Search seed catalog
                    mm_i = client.get_index(name="endee_multimodal_ui")
                    mm_results_raw = mm_i.query(vector=q_vec, top_k=1)
                    mm_results = [r for r in mm_results_raw if r.get('distance', 1.0) <= SIMILARITY_THRESHOLD]
                    
                    # 2. Search user uploads
                    user_mm_i = client.get_index(name="multimodal_kb")
                    user_mm_results_raw = user_mm_i.query(vector=q_vec, top_k=2)
                    user_mm_results = [r for r in user_mm_results_raw if r.get('distance', 1.0) <= SIMILARITY_THRESHOLD]
                    
                    # Show results
                    has_vis = False
                    if user_mm_results:
                        has_vis = True
                        for match in user_mm_results:
                            meta = match.get('meta', {})
                            src = meta.get('source')
                            path = f"uploads/{src}"
                            if os.path.exists(path):
                                if meta.get('type') == 'video':
                                    st.video(path)
                                else:
                                    st.image(path, caption=f"User Upload: {src}")

                    for m in mm_results:
                        has_vis = True
                        meta = m.get('meta', {})
                        st.image(meta.get('url'), caption=f"Catalog: {meta.get('desc')}")
                        
                    if not has_vis:
                        st.warning("🖼️ I have no such type exist in my database (Visual)")
                except:
                    st.caption("No visual search results.")

