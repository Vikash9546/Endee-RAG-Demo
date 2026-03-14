"""
query.py — RAG Retrieval & Generation Script
==============================================
1. Takes a user question
2. Converts it into a vector using Sentence-Transformers (same model from ingest)
3. Queries Endee to find the top 3 most similar text chunks
4. Sends those 3 chunks + the user question to an LLM:
   - Primary:  OpenAI GPT-4o-mini  (if OPENAI_API_KEY is set)
   - Fallback: Google Gemini        (if GEMINI_API_KEY is set)
   - Both work, pick whichever key you have.
"""

import os
import sys
import argparse
from sentence_transformers import SentenceTransformer
from endee import Endee

INDEX_NAME = "knowledge_base"


# ── Core Retrieval Function ─────────────────────────────

def retrieve(question: str, model, index, top_k: int = 3) -> list[dict]:
    """
    Takes a user question, converts it to a vector using the same
    embedding model from ingest, and queries Endee for top-k matches.
    """
    query_vector = model.encode([question])[0].tolist()
    results = index.query(vector=query_vector, top_k=top_k)
    return results or []


# ── LLM Generation Functions ────────────────────────────

def build_prompt(question: str, contexts: list[str]) -> str:
    """
    Build the RAG prompt:
    "Use the following context to answer the question: [Context]. Question: [User Query]"
    """
    context_block = "\n\n---\n\n".join(contexts)
    return (
        f"Use the following context to answer the question: "
        f"{context_block}. "
        f"Question: {question}"
    )


def generate_with_openai(prompt: str, api_key: str) -> str:
    """Generate answer using OpenAI GPT-4o-mini."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful technical knowledge assistant. Answer strictly based on the provided context."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
        max_tokens=600,
    )
    return response.choices[0].message.content


def generate_with_gemini(prompt: str, api_key: str) -> str:
    """Generate answer using Google Gemini (free tier). Tries multiple models."""
    import google.generativeai as genai
    import time as _time

    genai.configure(api_key=api_key)

    # Try multiple models — ordered by free tier generosity
    models_to_try = ["gemini-2.0-flash-lite", "gemini-1.5-flash", "gemini-2.0-flash"]
    last_error = None

    for model_name in models_to_try:
        for attempt in range(2):  # retry once per model
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                return response.text
            except Exception as e:
                last_error = e
                err_str = str(e)
                # Stop immediately for invalid/expired API key errors
                if "API_KEY_INVALID" in err_str or "API key expired" in err_str or "API key not valid" in err_str:
                    raise
                if "429" in err_str and attempt == 0:
                    _time.sleep(10)  # wait before retry on rate limit
                elif "404" in err_str:
                    break  # model doesn't exist, skip to next
                else:
                    break  # other error, skip to next model

    raise last_error


def generate_answer(question: str, contexts: list[str]) -> str:
    """
    Tries to generate an LLM answer using available API keys.
    Priority: OpenAI → Gemini → raw context fallback.
    """
    prompt = build_prompt(question, contexts)

    # Try OpenAI first
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        try:
            return generate_with_openai(prompt, openai_key)
        except Exception as e:
            print(f"  ⚠  OpenAI failed ({e}), trying Gemini...")

    # Try Gemini as fallback
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if gemini_key:
        try:
            return generate_with_gemini(prompt, gemini_key)
        except Exception as e:
            err_str = str(e)
            if "API_KEY_INVALID" in err_str or "API key expired" in err_str or "API key not valid" in err_str:
                print(f"  ⚠  Gemini API key is invalid or expired. Please renew: https://aistudio.google.com/apikey")
            else:
                print(f"  ⚠  Gemini failed: {e}")

    return None


# ── Main CLI ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Query the Endee Knowledge Base (Semantic Search + RAG)"
    )
    parser.add_argument("question", type=str, help="Your question in natural language")
    parser.add_argument("--top_k", type=int, default=3, help="Number of context chunks to retrieve (default: 3)")
    args = parser.parse_args()

    question = args.question

    print("╔══════════════════════════════════════════════════╗")
    print("║   Endee Knowledge Base — RAG Query Engine       ║")
    print("╚══════════════════════════════════════════════════╝")

    # ── Step 1: Load same embedding model ─────────────
    print(f"\n[1/3] Encoding question: \"{question}\"")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # ── Step 2: Connect to Endee & Retrieve ───────────
    print(f"[2/3] Querying Endee for top {args.top_k} matching chunks...")
    client = Endee()

    try:
        index = client.get_index(name=INDEX_NAME)
    except Exception:
        print("❌  Index not found. Run `python ingest.py` first.")
        return

    results = retrieve(question, model, index, top_k=args.top_k)

    if not results:
        print("⚠  No results found in Endee.")
        return

    # Display semantic search results
    print("\n" + "=" * 55)
    print("  🔍  SEMANTIC SEARCH RESULTS (from Endee)")
    print("=" * 55)

    contexts = []
    for i, match in enumerate(results):
        meta     = match.get("meta", {})
        text     = meta.get("text", "")
        source   = meta.get("source", "unknown")
        distance = match.get("distance", 0)

        contexts.append(text)
        print(f"\n  [{i+1}] {source}  (distance: {distance:.4f})")
        print(f"      {text[:200]}{'...' if len(text) > 200 else ''}")

    # ── Step 3: Send to LLM for Generation ────────────
    print("\n" + "=" * 55)
    print("  🤖  RAG GENERATION (LLM + Endee Context)")
    print("=" * 55)

    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    has_gemini = bool(os.environ.get("GEMINI_API_KEY"))

    if not has_openai and not has_gemini:
        print("\n  ℹ  No LLM API key found. Set one of these:")
        print("     export OPENAI_API_KEY='sk-...'")
        print("     export GEMINI_API_KEY='AIza...'   (free at https://aistudio.google.com/apikey)")
        return

    llm_name = "OpenAI GPT-4o-mini" if has_openai else "Google Gemini"
    print(f"\n  Generating answer via {llm_name} with Endee context...\n")

    answer = generate_answer(question, contexts)

    if answer:
        print("  ╔══════════════════════════════════════════════╗")
        print("  ║  ✨  FINAL ANSWER                           ║")
        print("  ╚══════════════════════════════════════════════╝")
        print(f"\n  {answer}\n")
    else:
        print("  ❌  All LLM providers failed. Showing raw context:")
        for ctx in contexts:
            print(f"  {ctx[:300]}")
            print()


if __name__ == "__main__":
    main()
