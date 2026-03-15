import os
from sentence_transformers import SentenceTransformer
from endee import Endee, Precision

INDEX_NAME = "endee_recommendations"

PRODUCTS = [
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

def main():
    print("Loading embedding model (all-MiniLM-L6-v2) for Recommendations...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = Endee()
    
    # 1. Setup Index
    try:
        client.delete_index(INDEX_NAME)
    except Exception:
        pass
    
    print(f"Creating Endee Index: '{INDEX_NAME}'")
    try:
        client.create_index(name=INDEX_NAME, dimension=384, space_type="cosine", precision=Precision.FLOAT32)
    except Exception as e:
        print("Index might already exist:", e)
        
    index = client.get_index(name=INDEX_NAME)
    
    # 2. Ingest Products
    print(f"Ingesting {len(PRODUCTS)} products into the recommendation engine...")
    embeddings = model.encode([p["desc"] for p in PRODUCTS])
    payloads = []
    
    for p, emb in zip(PRODUCTS, embeddings):
        payloads.append({
            "id": p["id"],
            "vector": emb.tolist(),
            "meta": p
        })
        
    index.upsert(payloads)
    
    # 3. Recommend Context
    user_interest = "I am looking for something to help me track my runs and workouts."
    print(f"\nUser Profile/Interest: '{user_interest}'")
    
    user_vec = model.encode([user_interest])[0].tolist()
    
    print("Finding recommendations in Endee Vector DB...")
    results = index.query(vector=user_vec, top_k=2)
    
    print("\n==================================================")
    print("🎯 TOP RECOMMENDATIONS")
    print("==================================================")
    if not results:
        print("No recommendations found.")
    else:
        for i, match in enumerate(results):
            meta = match.get("meta", {})
            distance = match.get("distance", "N/A")
            print(f"[{i+1}] {meta.get('desc')} (Category: {meta.get('category')})")
            print(f"    -> Similarity Score/Distance: {distance:.4f}")

if __name__ == "__main__":
    main()
