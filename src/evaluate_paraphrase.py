"""
Paraphrase-based retrieval evaluation.
Rephrases 50 questions using Claude, then checks if the original document
is still retrieved in top-5. More honest than exact self-retrieval.
"""

import os
import time
import pandas as pd
import anthropic
from pathlib import Path
from dotenv import load_dotenv
from retrieval import retrieve

load_dotenv(Path(__file__).parent.parent / ".env")

SAMPLE_CSV    = Path(__file__).parent.parent / "data" / "sample.csv"
EVAL_SIZE     = 50   # total questions to evaluate
TOP_K         = 5
PER_TASK      = EVAL_SIZE // 8   # ~6 per task type

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def paraphrase(question: str) -> str:
    """Ask Claude Haiku to rephrase the question differently."""
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=128,
        messages=[{
            "role": "user",
            "content": (
                f"Rephrase the following medical question in different words. "
                f"Keep the clinical meaning identical but change the wording. "
                f"Reply with only the rephrased question, nothing else.\n\n"
                f"Question: {question}"
            ),
        }],
    )
    return response.content[0].text.strip()


def run_evaluation():
    print("Loading sample data...")
    df = pd.read_csv(SAMPLE_CSV)

    # Stratified sample: up to PER_TASK rows per task type
    chunks = []
    for task, group in df.groupby("task"):
        chunks.append(group.sample(min(len(group), PER_TASK), random_state=42))
    sampled = pd.concat(chunks).head(EVAL_SIZE).reset_index(drop=True)

    print(f"Sampled {len(sampled)} questions across {sampled['task'].nunique()} task types")
    print(f"Task breakdown: {sampled['task'].value_counts().to_dict()}\n")

    hits        = 0
    misses      = []
    sim_scores  = []
    paraphrases = []

    print(f"Generating paraphrases + evaluating (top-{TOP_K})...\n")

    for i, row in sampled.iterrows():
        original = row["question"]

        rephrased = paraphrase(original)
        paraphrases.append({"original": original, "paraphrase": rephrased, "task": row["task"]})

        results       = retrieve(rephrased, top_k=TOP_K, min_similarity=0.0)  # no threshold for eval
        returned_qs   = [r["question"] for r in results]
        top_sim       = results[0]["similarity"] if results else 0.0
        sim_scores.append(top_sim)

        hit = original in returned_qs
        if hit:
            hits += 1
        else:
            misses.append({
                "original":   original[:80],
                "paraphrase": rephrased[:80],
                "top_result": returned_qs[0][:80] if returned_qs else "none",
                "top_sim":    top_sim,
                "task":       row["task"],
            })

        status = "HIT " if hit else "MISS"
        print(f"  [{i+1:2d}/{len(sampled)}] {status} | sim={top_sim:.3f} | {row['task']}")
        print(f"         orig : {original[:70]}...")
        print(f"         rephr: {rephrased[:70]}...")
        print()

        time.sleep(0.1)  # avoid rate limiting

    # --- Results ---
    hit_rate = hits / len(sampled) * 100
    avg_sim  = sum(sim_scores) / len(sim_scores) if sim_scores else 0

    print("=" * 60)
    print(f"Hit Rate      : {hits}/{len(sampled)} = {hit_rate:.1f}%")
    print(f"Avg Top-1 Sim : {avg_sim:.3f}")

    if misses:
        print(f"\nMisses ({len(misses)}):")
        for m in misses:
            print(f"  Task      : {m['task']}")
            print(f"  Original  : {m['original']}")
            print(f"  Paraphrase: {m['paraphrase']}")
            print(f"  Top match : {m['top_result']}  (sim={m['top_sim']:.3f})")
            print()

    # Per-task breakdown
    task_hits = {}
    for m in misses:
        task_hits[m["task"]] = task_hits.get(m["task"], 0)
    for _, row in sampled.iterrows():
        t = row["task"]
        if t not in task_hits:
            task_hits[t] = 0

    print("Per-task hit rate:")
    task_counts = sampled["task"].value_counts().to_dict()
    miss_counts = {}
    for m in misses:
        miss_counts[m["task"]] = miss_counts.get(m["task"], 0) + 1
    for task, total in task_counts.items():
        missed = miss_counts.get(task, 0)
        task_hit_rate = (total - missed) / total * 100
        print(f"  {task:<35} {total - missed}/{total} = {task_hit_rate:.0f}%")

    print("\nInterpretation:")
    if hit_rate >= 80:
        print("  Good — paraphrase retrieval is solid.")
    elif hit_rate >= 60:
        print("  Moderate — retrieval degrades with rephrasing.")
    else:
        print("  Weak — consider a stronger embedding model.")

    # Save paraphrases for reference
    out = Path(__file__).parent.parent / "data" / "paraphrase_eval.csv"
    pd.DataFrame(paraphrases).to_csv(out, index=False)
    print(f"\nParaphrases saved to {out}")

    return hit_rate


if __name__ == "__main__":
    run_evaluation()
