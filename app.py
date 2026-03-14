import os
import streamlit as st
from sentence_transformers import SentenceTransformer
from endee import Endee, Precision
import time

# --- Page Config ---
st.set_page_config(page_title="Endee AI Apps", page_icon="⚡", layout="wide")

# --- App State Cache ---
@st.cache_resource
def load_models():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model

@st.cache_resource
def get_client():
    return Endee()

# Initialize objects
embedding_model = load_models()
client = get_client()

# --- Helpers ---
def init_index(index_name):
    # Ensure index exists
    try:
        client.get_index(name=index_name)
    except:
        try:
            client.create_index(name=index_name, dimension=384, space_type="cosine", precision=Precision.FLOAT32)
        except:
            pass
    return client.get_index(name=index_name)

# --- UI Sidebar Setup ---
st.sidebar.title("Endee Use Cases")
app_mode = st.sidebar.selectbox("Choose the application:", 
    ["1. Semantic Search & RAG", "2. Recommendations", "3. Agentic Memory"])

st.sidebar.markdown("---")
st.sidebar.info("Endee is the high-performance Vector DB powering all these models synchronously under the hood.")

# ==========================================================
# APP 1: SEMANTIC SEARCH & RAG
# ==========================================================
if app_mode == "1. Semantic Search & RAG":
    st.title("📚 Semantic Search & RAG Pipeline")
    st.markdown("Query the knowledge base! *(Endee will find context based on meaning, not just exact keywords)*")
    
    # Check if index exists with data
    try:
        rag_index = client.get_index("endee_rag_demo")
        stats = "Ready! (Data exists in index)"
    except:
        rag_index = None
        stats = "Index missing or empty! Run `ingest.py` first."
        
    st.caption(f"Status: {stats}")
    
    query = st.text_input("Ask a question about the indexed knowledge:", "What is Endee built for?")
    top_k = st.slider("Number of context chunks to retrieve:", 1, 5, 3)
    
    if st.button("Search Meaning in Endee"):
        if not rag_index:
            st.error("Please ensure the `endee_rag_demo` index exists and is populated via `python ingest.py`")
        else:
            with st.spinner("Embedding query and fetching nearest neighbors..."):
                query_vec = embedding_model.encode([query])[0].tolist()
                
                start_time = time.time()
                results = rag_index.query(vector=query_vec, top_k=top_k)
                latency = (time.time() - start_time) * 1000
                
                st.success(f"Results retrieved in {latency:.2f} ms")
                
                if results:
                    st.markdown("### 🎯 Semantic Search Context Retrieved:")
                    for i, match in enumerate(results):
                        meta = match.get('meta', {})
                        score = match.get('distance', "N/A")
                        
                        with st.expander(f"Context {i+1} : Score {score:.4f} - {(meta.get('source', 'Unknown'))}"):
                            st.write(meta.get('text', 'No text metadata available.'))
                else:
                    st.warning("No relevant results found.")

# ==========================================================
# APP 2: RECOMMENDATIONS
# ==========================================================
elif app_mode == "2. Recommendations":
    st.title("🛍️ AI Recommendations Engine")
    st.markdown("Endee searches for similar products based on a natural language user profile description.")
    
    rec_index = init_index("endee_recommendations_ui")
    
    # Base catalog
    PRODUCTS = [
        {"id": "p1", "desc": "Wireless Noise Cancelling Headphones, over-ear, black", "category": "Electronics"},
        {"id": "p2", "desc": "Running Shoes, lightweight, breathable mesh, blue", "category": "Footwear"},
        {"id": "p3", "desc": "Yoga Mat, non-slip, eco-friendly cork, 5mm thick", "category": "Fitness"},
        {"id": "p4", "desc": "Smartwatch with Heart Rate Monitor and GPS, waterproof", "category": "Electronics"},
        {"id": "p5", "desc": "Organic Green Tea, loose leaf, 100g", "category": "Groceries"}
    ]
    
    # Load catalog into DB (quick operation for 5 items)
    with st.spinner("Ensuring product catalog is loaded into Endee..."):
        payloads = [{"id": p["id"], "vector": embedding_model.encode([p["desc"]])[0].tolist(), "meta": p} for p in PRODUCTS]
        rec_index.upsert(payloads)
        
    st.markdown("##### Current Vector Product Catalog:")
    st.table(PRODUCTS)
    
    user_interest = st.text_input("Describe your interests (Intent-based Search):", "I want to track my heart rate while exercising outside.")
    
    if st.button("Get Recommendations"):
        with st.spinner("Finding closest vectors in Endee..."):
            user_vec = embedding_model.encode([user_interest])[0].tolist()
            results = rec_index.query(vector=user_vec, top_k=2)
            
            st.markdown("### 🎯 Top Recommendations:")
            col1, col2 = st.columns(2)
            
            for i, match in enumerate(results):
                meta = match.get('meta', {})
                dist = match.get('distance', 0)
                
                with (col1 if i % 2 == 0 else col2):
                    st.info(f"**{meta.get('category', '')}**\n\n{meta.get('desc', '')}\n\n*Similarity: {dist:.4f}*")

# ==========================================================
# APP 3: AGENTIC MEMORY
# ==========================================================
elif app_mode == "3. Agentic Memory":
    st.title("🤖 Agentic Stateful Memory Example")
    st.markdown("Agents use Endee to persistently store past thoughts/actions and recall them when a similar trigger occurs.")
    
    agent_index = init_index("endee_agent_memory_ui")
    
    st.markdown("#### Step 1: Add a memory to the Agent")
    colA, colB = st.columns(2)
    with colA:
        new_thought = st.text_input("Agent's Event / Thought Context", "User asked about French.")
    with colB:
        new_action = st.text_input("Agent's Decided Action Routine", "set_language('fr')")
        
    if st.button("Log to Agent's Episodic Memory"):
        vec = embedding_model.encode([new_thought])[0].tolist()
        agent_index.upsert([{
            "id": f"mem_st_{int(time.time()*1000)}",
            "vector": vec,
            "meta": {"thought": new_thought, "action": new_action, "ts": time.time()}
        }])
        st.success(f"Remembered: '{new_thought}' -> Action: '{new_action}'")
        
    st.markdown("---")
    st.markdown("#### Step 2: Agent responds to new user context")
    new_context = st.text_input("New Task Assigned to Agent:", "User is inquiring about the weather in Paris.")
    
    if st.button("Recall Endee Memory"):
        vec = embedding_model.encode([new_context])[0].tolist()
        results = agent_index.query(vector=vec, top_k=1)
        
        if results:
            match = results[0]
            meta = match.get('meta', {})
            dist = match.get('distance', 0)
            
            st.markdown("### 💡 Agent Reaction")
            st.write(f"The closest past situation the agent remembered was: **'{meta.get('thought')}'** (Distance: {dist:.4f})")
            st.success(f"Agent will execute the pattern: `{meta.get('action')}`")
        else:
            st.warning("No related past memory logged in Endee.")
