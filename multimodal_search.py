"""
multimodal_search.py — Multi-Modal (Text & Image) Search Demo
================================================================
Demonstrates how Endee can replicate Pinterest visual search or medical imaging queries.
Uses the CLIP model (which maps text and images to the same vector space) to:
  1. Ingest images of products into Endee as vectors.
  2. Take a TEXT query ("a black t-shirt") and find the closest IMAGE vector in Endee.
"""

import os
import requests
from io import BytesIO
from PIL import Image
from sentence_transformers import SentenceTransformer
from endee import Endee, Precision

INDEX_NAME = "endee_multimodal_clip"
DIMENSION = 512  # The dimension size of clip-ViT-B-32 model

# ── 1. Create a Fake Image Catalog (Mock URLs for Demo) ─────

CATALOG = [
    {"id": "img_1", "category": "Fashion", "url": "https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?width=300", "desc": "A white plain t-shirt"},
    {"id": "img_2", "category": "Fashion", "url": "https://images.unsplash.com/photo-1503342217505-b0a15ec3261c?width=300", "desc": "A black plain t-shirt"},
    {"id": "img_3", "category": "Footwear", "url": "https://images.unsplash.com/photo-1542291026-7eec264c27ff?width=300", "desc": "Red Nike running shoes"},
    {"id": "img_4", "category": "Pets", "url": "https://images.unsplash.com/photo-1517849845537-4d257902454a?width=300", "desc": "A cute dog looking at camera"},
    {"id": "img_5", "category": "Vehicles", "url": "https://images.unsplash.com/photo-1492144534655-ae79c964c9d7?width=300", "desc": "A sleek silver sports car"}
]

def load_image(url: str) -> Image:
    """Helper function to download an image from a URL and format it for CLIP."""
    response = requests.get(url, stream=True)
    return Image.open(response.raw)

def main():
    print("==================================================================")
    print(" 📸 MULTIMODAL SEARCH: Text-to-Image Demo like Pinterest")
    print("==================================================================")
    
    print("\n[Admin] Loading CLIP Model (clip-ViT-B-32) which understands both Images AND Text...")
    model = SentenceTransformer("clip-ViT-B-32")
    
    # Initialize Endee Database
    client = Endee()
    
    try: client.delete_index(INDEX_NAME)
    except Exception: pass
    
    client.create_index(name=INDEX_NAME, dimension=DIMENSION, space_type="cosine", precision=Precision.FLOAT32)
    image_index = client.get_index(name=INDEX_NAME)

    # ── 2. Embed IMAGES into the Vector Database ────────────────
    print(f"\n[Admin] Downloading and encoding {len(CATALOG)} images into Endee vectors...")
    payloads = []
    
    for item in CATALOG:
        img = load_image(item["url"])
        
        # We pass an IMAGE to model.encode(), not text!
        img_vector = model.encode(img).tolist()
        
        payloads.append({
            "id": item["id"],
            "vector": img_vector,
            "meta": item
        })
        print(f"        ✓ Encoded Image: {item['desc']}")
        
    image_index.upsert(payloads)
    print("        ✓ All Images secured in Endee Long-Term memory.\n")

    # ── 3. Search using pure TEXT to find the IMAGE ─────────────
    
    # The user types text, but we search against the image vectors!
    user_query = "I want to buy a pair of athletic shoes for running."
    
    print("==================================================================")
    print(f"👤 USER TYPES TEXT QUERY: '{user_query}'")
    print("==================================================================")
    print("🔍 Searching Endee for the closest visual match...")
    
    query_vector = model.encode(user_query).tolist()
    
    # Query Endee for the closest images in the multidimensional space
    results = image_index.query(vector=query_vector, top_k=2)
    
    print("\n🎯 TOP VISUAL MATCHES RETURNED BY ENDEE:")
    if not results:
        print("No recommendations found.")
    else:
        for i, match in enumerate(results):
            meta = match.get("meta", {})
            distance = match.get("distance", "N/A")
            print(f"  [{i+1}] {meta.get('desc')} (Category: {meta.get('category')})")
            print(f"      -> Image URL: {meta.get('url')}")
            print(f"      -> Similarity Score/Distance: {distance:.4f}\n")
    
    print("✓ Endee successfully matched the textual intent to the Raw Image Vectors (like Pinterest)!")

if __name__ == "__main__":
    main()
