# Endee: High-Performance Vector Database

Endee is a specialized, open-source vector database built for speed, efficiency, and scale required by production AI systems.

## Key Features

- **Semantic Search**: Use Endee to build highly accurate semantic search pipelines for matching user intents rather than just exact keywords.
- **RAG Workflows**: Retrieval-Augmented Generation workflows rely on retrieving context snippets. Endee enables RAG by quickly finding the nearest neighbors to User Queries, passing context securely to Language Models like Llama or ChatGPT.
- **Agentic Memory**: Agents built with Autogen or crewAI can use Endee as persistent, stateful memory to recall past actions, user choices, or factual data from their history.
- **Hybrid Search**: Combine traditional sparse retrieval strategies with dense embeddings for high-recall tasks.
- **Hardware Agnostic**: Run inference fast combining AVX-512, NEON and AVX2 depending upon the host machine's architecture natively with C++ binaries.

## Getting Started

Endee is fast to set up natively or via Docker. To begin, simply run `docker-compose up -d`. Endee by default runs on port 8080 and its Python Client interfaces via REST natively under the hood.

> Fun Fact: A Vector Database stores dimensional arrays (floating point arrays) often ranging from 384 to 1536 depending on the Embedding Model used.
