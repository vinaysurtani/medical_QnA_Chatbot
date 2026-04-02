"""
Retrieval evaluation via hit rate.
Takes N questions already in Supabase, queries each one,
checks if the correct document comes back in top-k results.
"""

import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from supabase import create_client

load_dotenv(Path(__file__).parent.parent / ".env")

SAMPLE_CSV = Path(__file__).parent.parent / "data" / "sample.csv"
EVAL_SIZE  = 50   # number of questions to evaluate
TOP_K      = 5    # check if correct doc appears in top-k results


def run_evaluation():
    print("Loading sample data...")
    df = pd.read_csv(SAMPLE_CSV).head(EVAL_SIZE)

    print(f"Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Connecting to Supabase...")
    client = create_client(
        os.environ["SUPABASE_URL"],
        os.environ["SUPABASE_SERVICE_KEY"],
    )

    hits      = 0
    misses    = []
    scores    = []

    print(f"\nEvaluating {EVAL_SIZE} queries (top-{TOP_K})...\n")

    for i, row in df.iterrows():
        query     = row["question"]
        expected  = row["question"]   # the exact question should come back as top result

        embedding = model.encode(query, normalize_embeddings=True).tolist()
        response  = client.rpc(
            "match_documents",
            {"query_embedding": embedding, "match_count": TOP_K},
        ).execute()

        results          = response.data
        returned_qs      = [r["question"] for r in results]
        top_similarity   = results[0]["similarity"] if results else 0.0

        scores.append(top_similarity)

        if expected in returned_qs:
            hits += 1
        else:
            misses.append({
                "query":      query[:80],
                "top_result": returned_qs[0][:80] if returned_qs else "none",
                "similarity": top_similarity,
            })

    hit_rate = hits / EVAL_SIZE * 100
    avg_sim  = sum(scores) / len(scores)
    print(f"Hit Rate  : {hits}/{EVAL_SIZE} = {hit_rate:.1f}%")
    print(f"Avg Top-1 Similarity: {avg_sim:.3f}")

    if misses:
        print(f"\nMisses ({len(misses)}):")
        for m in misses[:5]:   # show first 5 misses
            print(f"  Query    : {m['query']}")
            print(f"  Top match: {m['top_result']}")
            print(f"  Sim score: {m['similarity']:.3f}")
            print()

    print("\nInterpretation:")
    if hit_rate >= 90:
        print("  Excellent — retrieval is very reliable.")
    elif hit_rate >= 70:
        print("  Good — retrieval works well for most queries.")
    elif hit_rate >= 50:
        print("  Moderate — consider tuning similarity threshold or embedding model.")
    else:
        print("  Poor — retrieval needs investigation.")

    return hit_rate


if __name__ == "__main__":
    run_evaluation()
