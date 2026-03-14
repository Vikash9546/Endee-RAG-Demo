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

---

## Setup and Execution Instructions

### 1. Start the Endee Vector Database
Before running the code, start **Endee** using the bundled `docker-compose.yml`:
```bash
docker-compose up -d
```
The Endee server will listen on `http://localhost:8080`.

### 2. Environment Setup
Install the necessary python dependencies using `pip`:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

*(Optional)* If you want to use the Generation capabilities (RAG), export your OpenAI API Key:
```bash
export OPENAI_API_KEY="sk-your-openai-key-here"
```

### 3. Ingest Documents into Endee
Add some `.txt` or `.md` files to the `data/` directory (you can copy your project files over to index them!). Run the script to ingest them into Endee:
```bash
python ingest.py
```
This script will parse your documents, generate embeddings, and store them inside the `endee_rag` index in your running Endee instance.

### 4. Query the System (Semantic Search + RAG)
Run the application script to chat with your knowledge base:
```bash
python query.py "What is Endee built for?"
```

If you configured an `OPENAI_API_KEY`, it will run full Generation (RAG) and print a coherent AI answer based on the search results. If not, it will default to outputting the pure Semantic Search retrieved context chunks.

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
