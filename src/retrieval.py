"""
Step 4: Retrieval layer.
Given a user query, embed it and find the top-k most similar documents
from Supabase using cosine similarity via pgvector.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from supabase import create_client

load_dotenv(Path(__file__).parent.parent / ".env")

_model  = None
_client = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _get_client():
    global _client
    if _client is None:
        _client = create_client(
            os.environ["SUPABASE_URL"],
            os.environ["SUPABASE_SERVICE_KEY"],
        )
    return _client


def retrieve(query: str, top_k: int = 5, min_similarity: float = 0.5) -> list[dict]:
    """
    Embed query and return top_k most similar documents from Supabase.
    Each result has: id, patient_id, task, question, answer, note, similarity.
    Results below min_similarity are discarded — prevents returning irrelevant
    sources for out-of-scope questions.
    """
    model = _get_model()
    embedding = model.encode(query, normalize_embeddings=True).tolist()

    client = _get_client()
    response = client.rpc(
        "match_documents",
        {"query_embedding": embedding, "match_count": top_k},
    ).execute()

    return [r for r in response.data if r["similarity"] >= min_similarity]


if __name__ == "__main__":
    # Quick smoke test
    test_queries = [
        "What medications was the patient discharged with?",
        "Does the patient have a history of diabetes?",
        "What was the patient's blood pressure on admission?",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        results = retrieve(query, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  [{i}] similarity={r['similarity']:.3f} | task={r['task']}")
            print(f"       Q: {r['question'][:80]}...")
            print(f"       A: {r['answer'][:80]}...")
