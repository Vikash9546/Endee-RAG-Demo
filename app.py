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
# ── Main Area Routing ────────────────────────────────────

st.title("🌐 Unified AI Search Dashboard")
st.markdown("One search powered by **Endee Vector DB**. Retrieves Knowledge, Products, and Visuals simultaneously.")

# Unified search input
if prompt := st.chat_input("Ask a question, find a product, or search visually..."):
    # ── 1. RAG Retrieve ──────────────────────────
    st.subheader("🤖 AI Knowledge Assistant")
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
        if gemini_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=gemini_key)
                gmodel = genai.GenerativeModel("gemini-1.5-flash")
                resp = gmodel.generate_content(f"Answer based on context: {context_block}\nQuestion: {prompt}")
                response_text = resp.text
            except: pass
        
        if not response_text:
            response_text = "⚠️ *Quota hit. Showing raw Endee Context:*\n\n" + contexts[0][:500] + "..."
        st.info(response_text)
    else:
        st.caption("No knowledge base results found.")

    st.divider()

    # ── 2. Recommendation & Visual Results ────────
    col_rec, col_vis = st.columns(2)

    with col_rec:
        st.subheader("🛍️ Product Recommendations")
        with st.spinner("Finding similar products..."):
            try:
                rec_i = client.get_index(name="endee_recommendations_ui")
                rec_results = rec_i.query(vector=model.encode([prompt])[0].tolist(), top_k=2)
                for m in rec_results:
                    meta = m.get('meta', {})
                    st.success(f"**{meta.get('category')}**\n\n{meta.get('desc')}")
            except:
                st.caption("No recommendations found. Use Sidebar to seed catalog.")

    with col_vis:
        st.subheader("📸 Visual Search")
        with st.spinner("Fetching visual matches..."):
            try:
                @st.cache_resource
                def load_clip(): return SentenceTransformer("clip-ViT-B-32")
                clip_m = load_clip()
                mm_i = client.get_index(name="endee_multimodal_ui")
                mm_results = mm_i.query(vector=clip_m.encode([prompt])[0].tolist(), top_k=1)
                for m in mm_results:
                    meta = m.get('meta', {})
                    st.image(meta.get('url'), caption=meta.get('desc'))
            except:
                st.caption("No visual matches. Use Sidebar to seed catalog.")
