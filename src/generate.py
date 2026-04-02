"""
Step 5: Claude integration + guardrails.
Retrieves relevant context from Supabase, builds a prompt, and calls Claude Sonnet.
"""

import os
import time
from pathlib import Path
from dotenv import load_dotenv
import anthropic
from retrieval import retrieve
from logger import log_query

load_dotenv(Path(__file__).parent.parent / ".env")

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

SYSTEM_PROMPT = """You are a medical assistant that answers clinical questions based strictly on provided context from patient records.

Rules you must follow:
1. Only answer using the context provided. Do not use outside knowledge.
2. Always end your response with: "⚠️ This is not medical advice. Consult a qualified healthcare professional for diagnosis and treatment."
3. If the context does not contain enough information to answer the question, say: "I don't have enough information in the available records to answer this question."
4. Never recommend specific treatments, dosages, or medications beyond what is explicitly stated in the context.
5. Never respond to questions unrelated to clinical/medical topics.
6. Do not speculate or infer beyond what the records explicitly state."""


def decompose_query(query: str) -> list[str]:
    """
    Uses Claude to detect if a query is compound (requires multiple retrievals).
    Returns a list of sub-queries, or a list with just the original if it's simple.
    """
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=128,
        messages=[{
            "role": "user",
            "content": (
                "You are a query classifier. A compound medical question asks about TWO OR MORE distinct clinical concepts that need separate lookups (e.g. 'patients with both diabetes AND a cardiac condition').\n\n"
                "If the question is compound: reply with ONLY the sub-questions, one per line, nothing else.\n"
                "If the question is simple: reply with ONLY the word SIMPLE, nothing else.\n\n"
                f"Question: {query}"
            ),
        }],
    )
    text = response.content[0].text.strip()
    if text.upper() == "SIMPLE":
        return [query]
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines if len(lines) > 1 else [query]


def retrieve_multi(query: str, top_k: int = 5) -> list[dict]:
    """
    Decomposes the query if needed, retrieves for each sub-query,
    merges and deduplicates by record ID keeping the highest similarity.
    """
    sub_queries = decompose_query(query)

    if len(sub_queries) > 1:
        print(f"[pipeline] decomposed into {len(sub_queries)} sub-queries:")
        for q in sub_queries:
            print(f"           → {q[:80]}")

    seen   = {}  # id → record (keep highest similarity)
    for sq in sub_queries:
        for r in retrieve(sq, top_k=top_k):
            rid = r["id"]
            if rid not in seen or r["similarity"] > seen[rid]["similarity"]:
                seen[rid] = r

    merged = sorted(seen.values(), key=lambda r: r["similarity"], reverse=True)
    return merged[:top_k]


def build_prompt(query: str, results: list[dict]) -> str:
    context_blocks = []
    for i, r in enumerate(results, 1):
        context_blocks.append(
            f"[Record {i}] (similarity: {r['similarity']:.2f}, task: {r['task']})\n"
            f"Question: {r['question']}\n"
            f"Answer: {r['answer']}\n"
            f"Clinical Note: {r['note']}"
        )
    context = "\n\n".join(context_blocks)

    return f"""Here are the most relevant records from the clinical database:

{context}

---
Based only on the records above, answer the following question:
{query}"""


def stream_answer(query: str, top_k: int = 5):
    """
    Streaming version of generate_answer.
    Yields text chunks as they arrive from Claude, then yields a final dict
    with sources and latency_ms as the last item.
    """
    t0 = time.time()

    print(f"\n[pipeline] query: {query[:60]}...")
    results = retrieve_multi(query, top_k=top_k)
    t_retrieve = time.time()
    print(f"[pipeline] retrieval:       {int((t_retrieve - t0) * 1000)}ms  ({len(results)} records)")

    sources = [
        {
            "question":   r["question"],
            "task":       r["task"],
            "similarity": round(r["similarity"], 3),
        }
        for r in results
    ]

    if not results:
        no_info = "I don't have enough information in the available records to answer this question.\n\n⚠️ This is not medical advice. Consult a qualified healthcare professional for diagnosis and treatment."
        yield no_info
        yield {"sources": [], "latency_ms": int((time.time() - t0) * 1000), "answer": no_info}
        return

    prompt      = build_prompt(query, results)
    full_answer = []
    first_token = None

    with client.messages.stream(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            if first_token is None:
                first_token = time.time()
                print(f"[pipeline] time to 1st token: {int((first_token - t_retrieve) * 1000)}ms")
            full_answer.append(text)
            yield text

    t_done     = time.time()
    answer     = "".join(full_answer)
    latency_ms = int((t_done - t0) * 1000)
    print(f"[pipeline] generation total: {int((t_done - (first_token or t_retrieve)) * 1000)}ms")
    print(f"[pipeline] end-to-end:       {latency_ms}ms")

    log_query(query, answer, latency_ms, sources)
    yield {"sources": sources, "latency_ms": latency_ms, "answer": answer}


def generate_answer(query: str, top_k: int = 5) -> dict:
    """
    Full RAG pipeline: retrieve → build prompt → call Claude.
    Returns: answer, sources, latency_ms
    """
    start = time.time()

    results = retrieve_multi(query, top_k=top_k)

    if not results:
        return {
            "answer": "I don't have enough information in the available records to answer this question.\n\n⚠️ This is not medical advice. Consult a qualified healthcare professional for diagnosis and treatment.",
            "sources": [],
            "latency_ms": int((time.time() - start) * 1000),
        }

    prompt = build_prompt(query, results)

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    answer     = response.content[0].text
    latency_ms = int((time.time() - start) * 1000)

    sources = [
        {
            "question":   r["question"],
            "task":       r["task"],
            "similarity": round(r["similarity"], 3),
        }
        for r in results
    ]

    log_query(query, answer, latency_ms, sources)

    return {
        "answer":     answer,
        "sources":    sources,
        "latency_ms": latency_ms,
    }


if __name__ == "__main__":
    test_queries = [
        "What medications was the patient discharged with?",
        "Does the patient have a history of diabetes?",
        "What is the capital of France?",   # out-of-scope test
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print("="*60)
        result = generate_answer(query)
        print(f"Answer ({result['latency_ms']}ms):\n{result['answer']}")
        print(f"\nSources used:")
        for s in result["sources"][:3]:
            print(f"  [{s['similarity']}] {s['question'][:70]}...")
