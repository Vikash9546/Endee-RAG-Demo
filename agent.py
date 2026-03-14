import sys
from sentence_transformers import SentenceTransformer
from endee import Endee, Precision
import time

INDEX_NAME = "endee_agent_memory"

def main():
    print("==================================================")
    print("🤖 Agentic AI Workflows: Stateful Memory Initializing")
    print("==================================================")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    print("Connecting to Endee Vector Database for Persistent Memory...")
    client = Endee()
    
    try:
        client.delete_index(INDEX_NAME)
    except Exception:
        pass
        
    try:
        client.create_index(name=INDEX_NAME, dimension=384, space_type="cosine", precision=Precision.FLOAT32)
    except Exception as e:
        print("Index might already exist:", e)
        
    index = client.get_index(name=INDEX_NAME)
    
    def remember(thought, action, memory_id=None):
        """Stores an episodic memory in Endee."""
        vec = model.encode([thought])[0].tolist()
        if not memory_id:
            memory_id = f"mem_{int(time.time()*1000)}"
            
        index.upsert([{
            "id": memory_id,
            "vector": vec,
            "meta": {"thought": thought, "action": action, "timestamp": time.time()}
        }])
        print(f"[Agent Saved Memory] Thought: '{thought}' -> Action: '{action}'")

    def recall(context, top_k=1):
        """Retrieves past memories relevant to the current context."""
        vec = model.encode([context])[0].tolist()
        results = index.query(vector=vec, top_k=top_k)
        if not results:
            return None
        return results[0] # Return the best match

    print("\n--- Simulating Past Agent Experiences ---")
    
    remember("User asked for the weather in Tokyo. I need to call the weather API.", "fetch_weather('Tokyo')", "mem_1")
    remember("User likes their responses translated to French.", "set_language('fr')", "mem_2")
    remember("The database password is historically passed via environment variables.", "os.getenv('DB_PASS')", "mem_3")
    
    time.sleep(1) # Allow slight delay to show timeline progress natively
    print("\n[New Agent Task]: 'Find the weather in Paris.'")
    
    # Agent tries to recall similar situations
    task = "Find the weather in Paris."
    print("Agent is recalling past context from Endee Memory...")
    past_memory_match = recall(task)
    
    if past_memory_match:
        past_memory = past_memory_match.get("meta", {})
        distance = past_memory_match.get("distance", "N/A")
        
        print("\n==================================================")
        print("💡 [Agent Memory Retrieved]")
        print("==================================================")
        print(f"Nearest past thought: '{past_memory.get('thought')}'")
        print(f"Similarity Distance:  {distance:.4f}")
        print(f"Chosen Action Route:  '{past_memory.get('action')}'")
    else:
        print("No relevant memory found.")
        
if __name__ == "__main__":
    main()
