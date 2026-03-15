# ⚡ Endee RAG Knowledge Base

A **Retrieval-Augmented Generation (RAG)** chatbot powered by [Endee](https://github.com/endee-io/endee) — a high-performance open-source vector database built for AI search.

Upload technical docs (PDFs, Markdown, Text), and the chatbot answers questions based **only** on those documents.

---

## Problem Statement

Traditional keyword search fails when users ask conceptual questions like *"How do I connect Prisma to MongoDB?"*. This project solves that with **semantic search** — it understands the *meaning* behind a query, retrieves the most relevant chunks from your documents stored in Endee, and feeds them to an LLM to generate a precise answer.

---

## How It Works

### The Python Workflow

| Step | What Happens | Tool Used |
|------|-------------|-----------|
| **1. Extract Text** | Read PDFs, Markdown, and `.txt` files | `PyMuPDF (fitz)` |
| **2. Chunking** | Split text into ~500-character overlapping pieces | Custom chunker |
| **3. Embeddings** | Convert each chunk into a 384-dim vector | `sentence-transformers/all-MiniLM-L6-v2` |
| **4. Store in Endee** | Upsert vectors + metadata into the vector database | `endee` Python SDK |
| **5. Retrieve** | User question → vector → find top 3 closest chunks in Endee | `Endee.query()` |
| **6. Generate** | Send chunks + question to LLM → get a grounded answer | `OpenAI GPT-4o-mini` |

```python
# Embedding example (Step 3)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
vector = model.encode("How to connect Prisma to MongoDB?")
```

### Architecture Diagram

```mermaid
graph TD
    A["📄 PDFs / Markdown / Text"] --> B["Python: PyMuPDF Text Extraction"]
    B --> C["Chunking (500 chars, 50 overlap)"]
    C --> D["Sentence-Transformers Embedding"]
    D -->|384-dim vectors| E[("⚡ Endee Vector Database")]
    
    F["❓ User Question"] --> G["Sentence-Transformers Embedding"]
    G -->|Query vector| E
    E -->|"Top 3 context chunks"| H["Prompt Assembly"]
    F --> H
    
    H --> I["🤖 OpenAI GPT-4o-mini"]
    I --> J["✅ Final Answer"]

    style E fill:#1a1a2e,stroke:#e94560,stroke-width:2px,color:#fff
    style J fill:#0f3460,stroke:#16213e,stroke-width:2px,color:#fff
```

---

## Quick Start (3 Commands)

### Prerequisites
- Python 3.10+
- Docker installed and running

### 1. Install dependencies
```bash
git clone https://github.com/Vikash9546/Endee-RAG-Demo.git
cd Endee-RAG-Demo
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
```

### 2. Start Endee + ingest sample docs
```bash
docker run -d -p 8080:8080 --name endee endeeio/endee-server:latest && python ingest.py
```

### 3. Query the knowledge base
```bash
# Option A: Use Google Gemini (FREE — get key at https://aistudio.google.com/apikey)
export GEMINI_API_KEY="gemini_api_key"
python query.py "What is Endee built for?"

# Option B: Use OpenAI GPT-4o-mini (paid)
export OPENAI_API_KEY="sk-your-key-here"
python query.py "What is Endee built for?"
```

> **How it works:** The script tries OpenAI first, then falls back to Gemini. If neither key is set, it still returns the raw semantic search results from Endee.

---

### Option 3: "Ghost-Protocol" (Agentic AI Workflow)
An AI agent that can "browse" its own memory to complete a multi-step task for **Automated Incident Response**.
```bash
python incident_agent.py
```
This runs an autonomous loop where the agent:
1. Receives a server crash error.
2. Queries Endee to see if a similar error happened before.
3. Automatically decides whether to **Fix** it (by executing a known solution) or **Escalate** it to a human based on difficulty. 

---

### Option 4: AI Recommendation System
A system that suggests products based on semantic similarity to a user's natural language profile.
```bash
python recommendation.py
```
This script:
1. Embeds a mock catalog of 5 products (e.g. Yoga Mats, Running shoes, Headphones, Smartwatches).
2. Takes a user profile like: *"I am looking for something to help me track my runs and workouts."*
3. Queries Endee and perfectly returns the **Running Shoes** and **Smartwatch** through semantic meaning.
4. **Logic**: Uses a similarity threshold to ensure random products aren't shown for unrelated queries.

---

---

### Bonus: Unified AI Search Dashboard
```bash
streamlit run app.py
```
Opens a high-performance **Multi-Feature Dashboard** at `http://localhost:8501`. 
- **🌐 Unified Search**: One search box for Knowledge and Products.
- **🤖 AI Knowledge Assistant**: Deep PDF/Text RAG using the new **Google Gemini 3-Flash**.
- **🛍️ Product recommendations**: Smart intent-based similarity filtering.

> **New Feature**: I have implemented a similarity threshold that guarantees the AI only shows relevant results. If nothing matches, the dashboard accurately tells you: *"I have no such type exist in my database."*

---

## How Endee Is Used

Endee is the **core component** of this application — it acts as the "memory" for the AI:

1. **Index Creation** — We create a vector index with `dimension=384`, `space_type="cosine"`, and `precision=FLOAT32` via `client.create_index()`.
2. **Upserting Vectors** — Document chunks are embedded and stored with `index.upsert()`, including metadata (source file, chunk text, chunk index).
3. **Querying** — User questions are embedded with the **same model** and sent to `index.query(vector=..., top_k=3)` for nearest-neighbor retrieval.
4. **RAG Pipeline** — The top 3 matching chunks from Endee are injected into the LLM prompt as grounding context, ensuring the AI answers based on *your* documents.

---

## Project Structure

```
endee-rag-demo/
├── app.py               # Streamlit chatbot UI (upload PDFs, ask questions)
├── ingest.py            # Ingestion pipeline: extract → chunk → embed → upsert
├── query.py             # CLI: semantic search + RAG generation
├── recommendation.py    # Bonus: product recommendation via vector similarity
├── agent.py             # Bonus: agentic AI memory pattern using Endee
├── data/                # Place your PDFs / Markdown / Text files here
│   └── endee_overview.md
├── docker-compose.yml   # One-command Endee server setup
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## Example Output

```
╔══════════════════════════════════════════════════╗
║   Endee Knowledge Base — RAG Query Engine       ║
╚══════════════════════════════════════════════════╝

[1/3] Encoding question: "What is Endee built for?"
[2/3] Querying Endee for top 3 matching chunks...

  🔍  SEMANTIC SEARCH RESULTS (from Endee)
=======================================================

  [1] data/endee_overview.md  (distance: 0.4544)
      # Endee: High-Performance Vector Database ...

  [2] data/endee_overview.md  (distance: 0.4669)
      Endee is a specialized, open-source vector database ...

  🤖  RAG GENERATION (LLM + Endee Context)
=======================================================

  ╔══════════════════════════════════════════════╗
  ║  ✨  FINAL ANSWER                           ║
  ╚══════════════════════════════════════════════╝

  Endee is built for speed, efficiency, and scale required
  by production AI systems. It supports semantic search,
  RAG workflows, agentic memory, and hybrid search...
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Vector Database | [Endee](https://github.com/endee-io/endee) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| PDF Extraction | `PyMuPDF` |
| LLM | OpenAI `gpt-4o-mini` |
| Frontend | Streamlit |
| Containerization | Docker |

---

## License

MIT
