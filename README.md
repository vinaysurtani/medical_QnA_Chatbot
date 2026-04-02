# Medical Q&A RAG Pipeline

A retrieval-augmented generation (RAG) system that answers clinical questions from patient discharge summaries. Built with sentence-transformers, Supabase pgvector, Claude Haiku, FastAPI, and Streamlit.

---

## What it does

Users ask clinical questions in a chat interface. The system retrieves the most relevant patient records from a vector database, passes them as context to Claude, and streams back a grounded answer with sources and a medical disclaimer.

Out-of-scope questions (non-medical, or below similarity threshold) are refused cleanly with no sources shown.

Compound queries ("patients with both diabetes AND a cardiac condition") are automatically decomposed into sub-queries, retrieved separately, and merged before generation.

---

## Architecture

```
User question
     │
     ▼
[Query Decomposition]  ← Claude Haiku detects compound queries, splits if needed
     │
     ▼
[Embedding]  ← sentence-transformers all-MiniLM-L6-v2 (384 dims, normalized)
     │
     ▼
[Vector Search]  ← Supabase pgvector cosine similarity, top-5, min_similarity=0.5
     │
     ▼
[Prompt Builder]  ← retrieved question + answer + full clinical note per record
     │
     ▼
[Claude Haiku]  ← streaming, context-only, medical disclaimer enforced
     │
     ▼
[FastAPI]  ← SSE streaming endpoint + JSON endpoint
     │
     ▼
[Streamlit UI]  ← chat interface, sources expander, session stats
     │
     ▼
[Supabase query_logs]  ← every query/response/latency logged
     │
     ▼
[Analytics Dashboard]  ← Streamlit dashboard reading from query_logs
```

---

## Dataset

**Source:** [MedS-Bench / clinical_notes.csv](https://huggingface.co/datasets/Henrychur/MMedBench) — 158,114 rows of clinical discharge summaries with paired Q&A.

**Columns:** `patient_id`, `note` (discharge summary), `question`, `answer`, `task`

**Task types (8):** Question Answering, Summarization, Named Entity Recognition, Relation Extraction, Paraphrasing, Coreference Resolution, Abbreviation Expansion, Temporal Information Extraction

**What we use:** A stratified 20,000-row sample (2,500 per task type). Only `question` is embedded. `answer` and `note` are passed as context to Claude at generation time.

---

## Project structure

```
medical_test_tool/
├── clinical_notes.csv          # raw dataset (not committed — download separately)
├── requirements.txt
├── supabase_setup.sql          # run once in Supabase SQL Editor
├── .env                        # secrets (not committed)
├── data/
│   ├── sample.csv              # 2k stratified sample
│   ├── sample_20k.csv          # 20k stratified sample (used for upload)
│   └── paraphrase_eval.csv     # saved paraphrases from evaluation
└── src/
    ├── sample_data.py          # generates sample CSVs from full dataset
    ├── embed_and_upload.py     # embeds questions, uploads to Supabase
    ├── retrieval.py            # retrieve(query, top_k, min_similarity)
    ├── generate.py             # RAG pipeline: decompose → retrieve → Claude
    ├── logger.py               # logs queries to Supabase query_logs
    ├── api.py                  # FastAPI: POST /ask, POST /ask/stream
    ├── app.py                  # Streamlit chat UI
    ├── dashboard.py            # Streamlit analytics dashboard
    ├── evaluate_retrieval.py   # exact self-retrieval hit rate (overfit baseline)
    └── evaluate_paraphrase.py  # paraphrase-based hit rate (honest eval)
```

---

## Setup

### 1. Prerequisites
- Python 3.10+
- A [Supabase](https://supabase.com) project (free tier works)
- An [Anthropic](https://console.anthropic.com) API key

### 2. Clone and install

```bash
git clone <your-repo-url>
cd medical_test_tool
python -m venv env
source env/bin/activate        # Windows: env\Scripts\activate
pip install -r requirements.txt
```

### 3. Environment variables

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=sk-ant-...
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-publishable-key
SUPABASE_SERVICE_KEY=your-service-role-key
```

Get keys from Supabase → Settings → API.
Use the **service role key** for `SUPABASE_SERVICE_KEY` — it bypasses Row Level Security for server-side writes.

### 4. Supabase setup

Run `supabase_setup.sql` in your Supabase SQL Editor. This creates:
- `pgvector` extension
- `documents` table with `embedding vector(384)`
- `match_documents()` RPC for cosine similarity search
- `query_logs` table

Then create the IVFFlat index for fast approximate search:

```sql
CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

### 5. Download the dataset

Download `clinical_notes.csv` from the dataset source and place it in the project root.

### 6. Generate sample and upload embeddings

```bash
cd medical_test_tool
source env/bin/activate
python src/sample_data.py          # generates data/sample_20k.csv
python src/embed_and_upload.py     # embeds + uploads 20k rows to Supabase
```

Upload takes ~10-15 minutes. If it fails mid-way, run `TRUNCATE TABLE documents;` in Supabase SQL Editor and re-run.

### 7. Run the app

**Terminal 1 — API:**
```bash
uvicorn src.api:app --reload
```

**Terminal 2 — Chat UI:**
```bash
streamlit run src/app.py
```

**Terminal 3 — Analytics (optional):**
```bash
streamlit run src/dashboard.py
```

Open `http://localhost:8501` for the chat app and `http://localhost:8000/docs` for the API docs.

---

## Evaluation

**Paraphrase-based hit rate: 77.1%** (48 questions, Claude Haiku paraphrases)

| Task | Hit Rate |
|------|----------|
| Question Answering | 100% |
| Relation Extraction | 100% |
| Temporal Information Extraction | 100% |
| Abbreviation Expansion | 83% |
| Coreference Resolution | 67% |
| Paraphrasing | 67% |
| Summarization | 67% |
| Named Entity Recognition | 33% |

NER is low because all NER questions share the same template ("What Named Entities related to the patient's...") — paraphrases retrieve a different patient's NER record with high similarity. This is a dataset artifact, not a retrieval failure. For real clinical Q&A use (Question Answering task), retrieval is 100%.

Run the evaluation yourself:
```bash
python src/evaluate_paraphrase.py
```

---

## Common blockers

### Supabase RLS error on insert
`new row violates row-level security policy`

You're using the publishable key (`SUPABASE_KEY`) for writes. Switch to `SUPABASE_SERVICE_KEY` in your `.env`. The service role key bypasses RLS.

### Upload fails mid-way with SSL error
`httpcore.ReadError: [SSL: SSLV3_ALERT_BAD_RECORD_MAC]`

Supabase dropped the connection — batch payload was too large. The script uses `BATCH_SIZE = 25` with 3 retries to handle this. If it still fails, `TRUNCATE TABLE documents;` in Supabase and re-run.

### Streamlit terminal filled with `torchvision` errors
These are warnings from Streamlit's file watcher scanning the `transformers` package. Not real errors — the app still works. Suppressed by `~/.streamlit/config.toml`:
```toml
[server]
fileWatcherType = "none"
```

### Sentence-transformers model slow on first query
The model loads lazily on the first request (~5s). Subsequent queries use the cached singleton and are much faster (~200ms for embedding).

### `KeyError: 'task'` in pandas groupby
Newer pandas versions drop the groupby key column in `apply()`. All groupby operations in this project use an explicit `for task, group in df.groupby("task"):` loop to avoid this.

### Port already in use (Streamlit on 8502, 8503...)
Previous Streamlit processes still running. Kill all at once:
```bash
pkill -f "streamlit run"
```

### Compound queries not decomposing
The query decomposition uses Claude Haiku. If it returns an explanation instead of sub-queries, check that the prompt in `decompose_query()` in `generate.py` is intact. The model should return `SIMPLE` for single-concept queries.

---

## Key design decisions

| Decision | Rationale |
|----------|-----------|
| Embed questions only | Consistent question-to-question comparison at query time. Notes are too long and structurally different from queries. |
| Supabase over ChromaDB | Cloud from day 1, single DB for vectors and logs, no local state to manage. |
| Claude Haiku over Sonnet | Lower latency (~3s vs ~8s generation), lower cost, sufficient for context-grounded QA where reasoning depth doesn't matter as much. |
| Streaming via SSE | Reduces perceived latency — user sees first tokens at ~2s instead of waiting 5-8s for full response. |
| Similarity threshold 0.5 | Filters out irrelevant sources for off-topic queries. Validated empirically — legitimate medical queries score 0.7+, irrelevant ones score below 0.5. |
| Query decomposition | Single-vector retrieval fails for AND queries. Decompose → retrieve per sub-query → merge handles compound clinical questions. |
| Full note to Claude | Original 500-char truncation risked missing key clinical details (medications, follow-up) often at the end of discharge summaries. Claude Haiku's 200k context window handles 5 full notes comfortably. |
