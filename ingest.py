import os
import glob
from sentence_transformers import SentenceTransformer
from endee import Endee, Precision

# Initialize Endee Client
# We use the correct endpoint from the documentation
client = Endee()

INDEX_NAME = "endee_rag_demo"

def chunk_text(text, chunk_size=500, overlap=50):
    """Simple text chunker based on characters."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    return chunks

def get_documentsFromDirectory(directory="data/"):
    """Reads all markdown and text files from the specified directory."""
    documents = []
    
    # Read Markdown and Text files
    files = glob.glob(f"{directory}/**/*.md", recursive=True) + glob.glob(f"{directory}/**/*.txt", recursive=True)
    
    print(f"Found {len(files)} files to process in {directory}.")
    
    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            # We use filename as title/id base
            filename = os.path.basename(file_path)
            
            # Chunk the content
            chunks = chunk_text(content)
            
            for i, chunk in enumerate(chunks):
                documents.append({
                    "id": f"{filename}-chunk-{i}",
                    "text": chunk,
                    "meta": {
                        "source": file_path,
                        "title": filename,
                        "chunk_index": i
                    }
                })
    return documents

def main():
    print("Loading Sentence Transformer Model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2") # 384 dimensions
    
    print("Connecting to Endee Vector DB...")
    
    # Recreate the Index for the Demo
    try:
        # Check if the index already exists. 
        # Using list index from API endpoint we saw in getting_started
        indexes = dict(client.list_indexes())
        
        # If the python client `list_indexes` returns list of dicts or just check via try except
    except Exception as e:
        pass
        
    try:
        # Note: Depending on the specific client version it may throw exception or overwrite
        # We will attempt to drop it if it exists to keep demo repeatable
        try:
             client.delete_index(INDEX_NAME)
        except:
             pass
    except Exception as e:
        print("Could not delete index (might not exist):", e)

    print(f"Creating Endee Index: '{INDEX_NAME}' with Dimension: 384")
    client.create_index(
        name=INDEX_NAME, 
        dimension=384, 
        space_type="cosine", 
        precision=Precision.FLOAT32 # Better precision for text
    )
    
    index = client.get_index(name=INDEX_NAME)
    
    documents = get_documentsFromDirectory()
    
    if not documents:
        print("No documents found in data/ directory. Please add some .txt or .md files!")
        return
        
    print(f"Chunked into {len(documents)} snippets. Generating Embeddings...")
    
    # Extract just text for embeddings
    texts = [doc["text"] for doc in documents]
    embeddings = model.encode(texts, show_progress_bar=True)
    
    print("Uploading to Endee Vectors...")
    
    # Format according to Endee `upsert` API
    payloads = []
    for doc, emb in zip(documents, embeddings):
        payloads.append({
             "id": doc["id"],
             "vector": emb.tolist(),
             "meta": {
                  "text": doc["text"],
                  "source": doc["meta"]["source"],
                  "title": doc["meta"]["title"]
             }
        })
        
    # Upsert with a reasonable batch size
    batch_size = 100
    for i in range(0, len(payloads), batch_size):
        batch = payloads[i:i+batch_size]
        try:
             index.upsert(batch)
             print(f"Upserted items {i} to {i+len(batch)} into '{INDEX_NAME}'.")
        except Exception as e:
             print(f"Failed to upsert batch: {e}")
             
    print("Data Ingestion to Endee Complete! Ready for querying.")

if __name__ == "__main__":
    main()
