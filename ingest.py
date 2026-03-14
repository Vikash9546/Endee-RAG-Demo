"""
ingest.py — AI-ML Knowledge Base Ingestion Pipeline
=====================================================
Reads PDFs (.pdf), Markdown (.md), and Text (.txt) files from the data/ folder,
chunks them into ~500 character pieces, converts each chunk into a 384-dim vector
using sentence-transformers, and upserts everything into the Endee vector database.
"""

import os
import glob
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from endee import Endee, Precision

# ── Config ──────────────────────────────────────────────
INDEX_NAME   = "knowledge_base"
DATA_DIR     = "data/"
CHUNK_SIZE   = 500   # characters per chunk
CHUNK_OVERLAP = 50   # overlap between consecutive chunks
DIMENSION    = 384   # all-MiniLM-L6-v2 output dimension

# ── Step 1: Extract Text ────────────────────────────────

def extract_text_from_pdf(filepath):
    """Use PyMuPDF to extract all text from a PDF file."""
    doc = fitz.open(filepath)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def extract_text_from_file(filepath):
    """Read plain text / markdown files."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def load_documents(directory=DATA_DIR):
    """Scan the data/ directory for PDFs, Markdown, and Text files."""
    documents = []

    # Collect all supported file types
    patterns = {
        "pdf": glob.glob(f"{directory}/**/*.pdf", recursive=True),
        "md":  glob.glob(f"{directory}/**/*.md",  recursive=True),
        "txt": glob.glob(f"{directory}/**/*.txt", recursive=True),
    }

    for filetype, files in patterns.items():
        for fp in files:
            if filetype == "pdf":
                text = extract_text_from_pdf(fp)
            else:
                text = extract_text_from_file(fp)
            documents.append({"path": fp, "text": text})

    return documents

# ── Step 2: Chunking ────────────────────────────────────

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping pieces of ~500 characters each."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks

# ── Step 3 + 4: Embed & Store in Endee ──────────────────

def main():
    # ── Load embedding model ──────────────────────────
    print("╔══════════════════════════════════════════════════╗")
    print("║   Endee Knowledge Base — Ingestion Pipeline     ║")
    print("╚══════════════════════════════════════════════════╝")

    print("\n[1/4] Loading Sentence-Transformers model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # ── Connect to Endee ──────────────────────────────
    print("[2/4] Connecting to Endee Vector Database...")
    client = Endee()

    # Recreate the index for a clean demo run
    try:
        client.delete_index(INDEX_NAME)
    except Exception:
        pass

    client.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        space_type="cosine",
        precision=Precision.FLOAT32,
    )
    index = client.get_index(name=INDEX_NAME)

    # ── Extract text from data/ ───────────────────────
    print(f"[3/4] Scanning '{DATA_DIR}' for .pdf / .md / .txt files...")
    documents = load_documents()

    if not documents:
        print("⚠  No documents found. Add files to the data/ folder and re-run.")
        return

    print(f"      Found {len(documents)} file(s).")

    # ── Chunk → Embed → Upsert ────────────────────────
    all_payloads = []
    for doc in documents:
        filename = os.path.basename(doc["path"])
        chunks = chunk_text(doc["text"])
        print(f"      • {filename}: {len(chunks)} chunks")

        texts = [c for c in chunks]
        vectors = model.encode(texts, show_progress_bar=False)

        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            all_payloads.append({
                "id": f"{filename}::chunk-{i}",
                "vector": vec.tolist(),
                "meta": {
                    "text": chunk,
                    "source": doc["path"],
                    "title": filename,
                    "chunk_index": i,
                },
            })

    print(f"\n[4/4] Upserting {len(all_payloads)} vectors into Endee index '{INDEX_NAME}'...")

    BATCH = 100
    for i in range(0, len(all_payloads), BATCH):
        batch = all_payloads[i : i + BATCH]
        index.upsert(batch)

    print(f"\n✅  Done! {len(all_payloads)} chunks indexed. Ready for queries.")


if __name__ == "__main__":
    main()
