# Endee RAG System

This project demonstrates a fully functional **Retrieval-Augmented Generation (RAG)** pipeline and **Semantic Search Engine** using [Endee](https://github.com/endee-io/endee) as its core vector database.

## Problem Statement
Building scalable AI search and RAG platforms requires a robust and exceptionally fast Vector Database to efficiently handle embeddings matching. This project shows how to ingest documents (like code or documentation), generate text embeddings, and construct semantic search and generative workflows on top of Endee's high-performance indexing.

## Features
- **Semantic Search**: Quickly index and retrieve document chunks based on conceptual similarity using `sentence-transformers` and Endee.
- **RAG Workflows (Retrieval Augmented Generation)**: Integrate an LLM (using the OpenAI API) to generate well-informed answers grounded linearly on the contexts retrieved from Endee.
- **Recommendations**: Analyze user intent and interests to fetch nearest-neighbor matches of a product or movie catalog in vector space natively.
- **Agentic AI Workflows**: Leverage Endee dynamically to simulate long-term stateful episodic memory for an autonomous AI agent to use in its execution loops.
- **Automated Ingestion Pipeline**: Contains an ingestor that splits large Markdown or Text files into clean chunks for indexing.
- **Local Deployment**: Includes a `docker-compose.yml` configuration to run the local Endee instance smoothly alongside the Python client.

## System Design and Technical Approach
1. **Embedding Generation**: We use `sentence-transformers/all-MiniLM-L6-v2` locally to compute 384-dimensional dense vectors for document chunks.
2. **Vector Database**: Endee handles storing and querying these embeddings. We configure a COSINE distance space with FP32 precision.
3. **Chunking & Ingestion**: Documents are broken into overlapping chunks of configurable size. These are packaged as JSON payloads and sent to `Endee.upsert()`.
4. **Retrieval**: User queries are embedded via the exact same model and sent to the `Endee.query()` API for K-nearest neighbor search.
5. **Generation**: The top `K` results are concatenated and injected into an LLM prompt as context to generate intelligent, well-informed answers.

### Architecture Flow

```mermaid
graph TD
    A[Raw Documents (.txt/.md)] --> B[Python Chunking Script]
    B --> C[Sentence-Transformers Model]
    C -->|Embeddings| D[(Endee Vector Database)]
    
    E[User Question] --> F[Sentence-Transformers Model]
    F -->|Query Embedding| D
    D -->|Top 3 Matches Context| G[Prompt Assembly]
    E --> G
    
    G --> H[OpenAI / Local LLM generation]
    H --> I[Final User Answer]
```

---

## Quick Start (Run it in 3 commands!)

**1. Create your environment & install dependencies:**
```bash
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
```

**2. Start Endee in the background and ingest the context data:**
```bash
docker-compose up -d && python ingest.py
```

**3. Test the RAG Pipeline by querying for information:**
*(Make sure to set your `OPENAI_API_KEY` to run the ChatGPT Generation capability!)*
```bash
export OPENAI_API_KEY="sk-your-openai-key"
python query.py "What is Endee built for?"
```

*(You can also run `streamlit run app.py` to open the fully fledged Graphical Interface that holds all AI demos!)*

### 4. Running the Interactive Streamlit UI App
To view **all 3 use cases** (Semantic Search/RAG, Recommendations, and Agentic Memory) in a single **graphical user interface (GUI)**, run the included `app.py` script:
```bash
python -m pip install streamlit
streamlit run app.py
```
This will open a dashboard in your browser (`http://localhost:8501`) where you can interactively test Endee's vector lookup capabilities spanning the three distinct AI capabilities!

### 5. Running the Recommendation Engine

See how Endee powers semantic similarity-based content matching by running the pre-loaded inventory demo:
```bash
python recommendation.py
```

### 6. Testing Agentic Workflows
You can test the Agentic Memory recall pattern (which mimics a LangChain / AutoGen AI Agent using Endee vector records to pick an action based on past tasks):
```bash
python agent.py
```
