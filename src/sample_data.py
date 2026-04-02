"""
Step 2: Load clinical_notes.csv, explore task distribution,
and create a stratified 2000-row sample saved to data/sample.csv
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

RAW_CSV = Path(__file__).parent.parent / "clinical_notes.csv"
SAMPLE_CSV    = DATA_DIR / "sample.csv"
SAMPLE_CSV_20K = DATA_DIR / "sample_20k.csv"
SAMPLE_SIZE   = 2000
SAMPLE_SIZE_20K = 20000


def load_and_explore(path: Path) -> pd.DataFrame:
    print("Loading CSV (this may take a moment for a large file)...")
    df = pd.read_csv(path)
    print(f"\nShape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nTask distribution:\n{df['task'].value_counts()}")
    print(f"\nNote length stats (chars):\n{df['note'].str.len().describe()}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    return df


def stratified_sample(df: pd.DataFrame, n: int = SAMPLE_SIZE) -> pd.DataFrame:
    task_counts = df["task"].value_counts()
    n_tasks = len(task_counts)
    per_task = n // n_tasks
    print(f"\n{n_tasks} task types found, sampling ~{per_task} rows each")

    parts = []
    for task, group in df.groupby("task"):
        parts.append(group.sample(min(len(group), per_task), random_state=42))
    sampled = pd.concat(parts)

    # top up to exactly n if rounding left us short
    if len(sampled) < n:
        remaining = df.drop(sampled.index).sample(n - len(sampled), random_state=42)
        sampled = pd.concat([sampled, remaining])

    return sampled.reset_index(drop=True)


if __name__ == "__main__":
    df = load_and_explore(RAW_CSV)
    sample = stratified_sample(df)
    sample.to_csv(SAMPLE_CSV, index=False)
    print(f"\nSample saved to {SAMPLE_CSV} ({len(sample)} rows)")
    print(f"Sample task distribution:\n{sample['task'].value_counts()}")

    sample_20k = stratified_sample(df, n=SAMPLE_SIZE_20K)
    sample_20k.to_csv(SAMPLE_CSV_20K, index=False)
    print(f"\n20k sample saved to {SAMPLE_CSV_20K} ({len(sample_20k)} rows)")
    print(f"20k task distribution:\n{sample_20k['task'].value_counts()}")
