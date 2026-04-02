"""
Step 3: Embed questions from data/sample.csv and upsert into Supabase documents table.
Run AFTER executing supabase_setup.sql in your Supabase SQL Editor.
"""

import os
import time
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from supabase import create_client

load_dotenv(Path(__file__).parent.parent / ".env")

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_KEY"]  # service key bypasses RLS for server-side writes
SAMPLE_CSV   = Path(__file__).parent.parent / "data" / "sample_20k.csv"
MODEL_NAME   = "all-MiniLM-L6-v2"
BATCH_SIZE   = 25


def embed_and_upload():
    print("Loading sample data...")
    df = pd.read_csv(SAMPLE_CSV)
    print(f"  {len(df)} rows loaded")

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print("Embedding questions...")
    embeddings = model.encode(
        df["question"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,   # cosine similarity via dot product
    )
    print(f"  Embeddings shape: {embeddings.shape}")

    print("Connecting to Supabase...")
    client = create_client(SUPABASE_URL, SUPABASE_KEY)

    print(f"Uploading in batches of {BATCH_SIZE}...")
    total = len(df)
    for start in range(0, total, BATCH_SIZE):
        batch_df  = df.iloc[start : start + BATCH_SIZE]
        batch_emb = embeddings[start : start + BATCH_SIZE]

        rows = [
            {
                "patient_id": row["patient_id"],
                "task":       row["task"],
                "question":   row["question"],
                "answer":     row["answer"],
                "note":       row["note"],
                "embedding":  emb.tolist(),
            }
            for (_, row), emb in zip(batch_df.iterrows(), batch_emb)
        ]

        for attempt in range(3):
            try:
                client.table("documents").insert(rows).execute()
                break
            except Exception as e:
                if attempt == 2:
                    raise
                print(f"  Retry {attempt + 1}/3 after error: {e}")
                time.sleep(2)
        print(f"  Uploaded rows {start + 1}–{min(start + BATCH_SIZE, total)}")

    print(f"\nDone. {total} rows uploaded to Supabase.")


if __name__ == "__main__":
    embed_and_upload()
