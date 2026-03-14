import os
import sys
import argparse
from sentence_transformers import SentenceTransformer
from endee import Endee
from openai import OpenAI

INDEX_NAME = "endee_rag_demo"

def main():
    parser = argparse.ArgumentParser(description="Query Endee Vector DB for Semantic Search & RAG")
    parser.add_argument("query", type=str, help="The search query or question.")
    parser.add_argument("--top_k", type=int, default=3, help="Number of semantic results to retrieve.")
    args = parser.parse_args()
    
    query_text = args.query
    print(f"Loading Sentence Transformer Model (all-MiniLM-L6-v2) to encode: '{query_text}'")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Generate Embedding for the query
    query_vector = model.encode([query_text])[0]
    
    print(f"\nSearching Endee Vector DB for top {args.top_k} results...")
    
    # Initialize Endee Client
    client = Endee()
    
    try:
        index = client.get_index(name=INDEX_NAME)
    except Exception as e:
        print(f"Error accessing index '{INDEX_NAME}'. Did you run ingest.py first?")
        return
        
    # Query Endee
    try:
         # Depending on client version, arguments to query could be list of vectors or a single vector
         results = index.query(vector=query_vector.tolist(), top_k=args.top_k)
    except Exception as e:
         print(f"Query Failed: {e}")
         return
         
    # Check if results exist
    if not results:
        print("No results found in Endee.")
        return
        
    print("\n" + "="*50)
    print("🎯 ENDEE SEMANTIC SEARCH RESULTS:")
    print("="*50)
    
    contexts = []
    
    # Iterate and print top matches
    for i, match in enumerate(results):
        # Distance metric (Cosine usually ranges between -1 and 1)
        score = match.get("distance", "N/A")
        
        # Meta dictionary where we stored our content
        meta = match.get("meta", {})
        text_chunk = meta.get("text", "No text found in meta.")
        source = meta.get("source", "Unknown Source")
        
        contexts.append(text_chunk)
        
        print(f"[{i+1}] Source: {source} (Score: {score})")
        print(f"Preview: {text_chunk[:150]}...")
        print("-" * 30)
        
    # Optional RAG Generation Step
    print("\n" + "="*50)
    print("🤖 RAG GENERATION (Retrieval-Augmented Generation)")
    print("="*50)
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Note: OPENAI_API_KEY environment variable not set. Skipping the LLM Generation step.")
        print("To enable full RAG, run: export OPENAI_API_KEY='sk-your-key'")
        return
        
    client_llm = OpenAI(api_key=api_key)
    
    context_str = "\n\n".join(contexts)
    
    prompt = f"Use the following context to answer the question: {context_str}. Question: {query_text}"
    
    print("Generating response via OpenAI LLM with Endee Context...")
    try:
         response = client_llm.chat.completions.create(
             model="gpt-4o-mini",
             messages=[
                 {"role": "system", "content": "You are a helpful and intelligent knowledge assistant."},
                 {"role": "user", "content": prompt}
             ],
             temperature=0.7,
             max_tokens=500
         )
         
         print("\n✨ FINAL ANSWER:")
         print(response.choices[0].message.content)
         
    except Exception as e:
         print(f"LLM Generation Failed: {e}")

if __name__ == "__main__":
    main()
