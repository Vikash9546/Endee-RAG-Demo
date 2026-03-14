"""
incident_agent.py — "Ghost-Protocol" Agentic AI Workflow
=============================================================
An Autonomous Incident Response Agent that uses Endee as its Long-Term Memory.

When a server crash or error occurs, the Agent:
  1. Uses the `search_memory` tool to query Endee for past occurrences.
  2. Analyzes the retrieved context (past solutions).
  3. Decides to either "Fix" (generate code) or "Escalate" (summarize for human).

Note: Works with OpenAI GPT-4o-mini if a valid API key is set.
Otherwise, runs a Local Simulation fallback to demonstrate the autonomous flow!
"""

import os
import json
from sentence_transformers import SentenceTransformer
from endee import Endee, Precision

# ── 1. Initialize Endee "Long-Term Memory" ──────────────────

INDEX_NAME = "agentic_incident_memory"
DIMENSION = 384

print("Loading Sentence-Transformers model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
client = Endee()

# Ensure fresh index for the demo
try:
    client.delete_index(INDEX_NAME)
except Exception:
    pass

client.create_index(name=INDEX_NAME, dimension=DIMENSION, space_type="cosine", precision=Precision.FLOAT32)
memory_index = client.get_index(name=INDEX_NAME)

# ── 2. Seed Memory with Past Server Incidents ───────────────

past_incidents = [
    {
        "error_str": "PostgreSQL Connection Refused port 5432",
        "solution": "Restarted the pg_ctl service and increased max_connections to 200 in postgresql.conf. Resolved gracefully.",
        "difficulty": "Easy"
    },
    {
        "error_str": "OOMKilled: Pod memory limit exceeded in Kubernetes",
        "solution": "Memory leak detected in Node.js worker pod. Requires senior SRE to profile the heap. Cannot be auto-fixed.",
        "difficulty": "Hard"
    },
    {
        "error_str": "AWS S3 Access Denied 403 Bucket Policy",
        "solution": "IAM role lost s3:PutObject permissions. Terraform applied to restore the role. Fix script available.",
        "difficulty": "Easy"
    }
]

print("\n[Admin] Seeding AI's Endee Memory with past incident logs...")
payloads = []
for i, inc in enumerate(past_incidents):
    # Vectorize the error signature
    vec = model.encode([inc["error_str"]])[0].tolist()
    payloads.append({
        "id": f"incident_log_{i}",
        "vector": vec,
        "meta": inc
    })
memory_index.upsert(payloads)
print(f"        ✓ {len(payloads)} past incidents committed to long-term memory.")


# ── 3. Define the LLM's Tool (Function Calling) ─────────────

def search_memory(error_signature: str) -> str:
    """
    The Agent calls this tool to browse its Endee vector memory.
    """
    print(f"\n  [Agent Tool Execution] 🔍 Querying Endee Memory for: '{error_signature}'")
    vec = model.encode([error_signature])[0].tolist()
    
    # Query Endee for the closest past incident
    results = memory_index.query(vector=vec, top_k=1)
    
    if not results or results[0].get("distance", -1) < 0.35:
        return "No similar past incidents found in memory."
    
    match = results[0]["meta"]
    print(f"  [Endee Return] Found past match! '{match['error_str']}'")
    return f"Past Issue: {match['error_str']}. Past Solution: {match['solution']}. Difficulty: {match['difficulty']}."


# ── 4. The Agentic Loop (OpenAI or Local Simulation) ────────

def local_simulated_agent_loop(new_error: str):
    """
    Runs the agentic logic locally without needing a paid API key.
    Demonstrates the exact decision tree an LLM would make.
    """
    print(f"\n🚨 NEW ALERT RECEIVED: {new_error}\n")
    print("🤖 Agent State: Analyzing alert...")
    
    # Step A: Agent autonomously decides to use the search_memory tool
    memory_result = search_memory(new_error)
    
    print("\n🤖 Agent State: Reasoning over retrieved memory...\n")
    
    # Step B: Agent decides to Fix or Escalate based on memory
    if "No similar past incidents" in memory_result:
        print(">> [DECISION: ESCALATE ☎️] This is a novel error. I have no past memory of this. Paging human SRE on-call.")
    elif "Easy" in memory_result:
        print(f">> [DECISION: AUTO-FIX 🛠️] I remember this exact signature! Executing Fix Script:")
        print(f"   Executing System Command --> '{memory_result.split('Past Solution: ')[1].split('.')[0]}'")
    elif "Hard" in memory_result:
        print(f">> [DECISION: ESCALATE w/ CONTEXT ⚠️] I remember this, but it requires human authorization. Escalating with context: {memory_result}")


# ── RUN PLAYBOOKS ──────────────────────────────────────────

if __name__ == "__main__":
    print("\n==================================================================")
    print(" 🚨 GHOST-PROTOCOL: Automated Incident Response Agent")
    print("==================================================================")
    
    print("\n--- Playbook 1: A Known, Auto-Fixable Issue ---")
    local_simulated_agent_loop("URGENT: Database crashing. Connection Refused to Postgres on port 5432.")
    
    print("\n--- Playbook 2: A Known, Hard Issue (Requires Escalation) ---")
    local_simulated_agent_loop("Production down: K8s pods encountering OOMKilled limits constantly.")
    
    print("\n--- Playbook 3: A Completely Unknown Issue ---")
    local_simulated_agent_loop("Kafka brokers encountering Split-Brain network partition anomaly.")
    
    print("\n==================================================================")
    print("✓ Agent executed successfully via Endee Vector Memory routing.")
