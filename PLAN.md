# Medical Chatbot Pipeline — Build Plan

## Architecture
- **Data**: `clinical_notes.csv` — 158k rows, columns: `patient_id`, `note`, `question`, `answer`, `task`
- **Retrieval**: embed `question` field using `sentence-transformers`, store vectors in Supabase pgvector
- **Generation**: retrieve top-k similar Q&A pairs → pass as context to Claude Sonnet → answer
- **Backend**: FastAPI
- **Frontend**: Streamlit
- **DB**: Supabase (pgvector for embeddings, Postgres for query logs)

---

## Steps

### [DONE] Step 1 — Project Setup
- Created folder structure: `src/`, `data/`, `logs/`, `tests/`
- Created `requirements.txt`, installed all dependencies into `env/`
- `.env` has: `ANTHROPIC_API_KEY`, `SUPABASE_URL`, `SUPABASE_KEY`

### [DONE] Step 2 — Data Sampling (`src/sample_data.py`)
- Loaded full CSV (158,114 rows, 8 task types, avg note ~1938 chars)
- Stratified sample: 250 rows × 8 tasks = **2000 rows** → `data/sample.csv`

### [DONE] Step 3 — Embeddings + Supabase Setup (`src/embed_and_upload.py`)
- Enable `pgvector` extension in Supabase
- Create `documents` table: `id`, `patient_id`, `task`, `question`, `answer`, `note`, `embedding`
- Embed all 2000 questions using `sentence-transformers` (model: `all-MiniLM-L6-v2`)
- Upsert rows + embeddings into Supabase

### [DONE] Step 4 — Retrieval Layer (`src/retrieval.py`)
- `retrieve(query, top_k=5, min_similarity=0.3)` — embed query, cosine similarity search via Supabase pgvector
- Similarity threshold filters out irrelevant results (no sources shown for out-of-scope questions)
- Paraphrase eval (`src/evaluate_paraphrase.py`): 77.1% hit rate across 48 questions
  - Strong: Question Answering 100%, Relation Extraction 100%, Temporal 100%
  - Weak: NER 33% — structural near-duplicates in dataset, not a real retrieval failure
  - Conclusion: retrieval is solid for the actual use case (QA tasks)

### [DONE] Step 5 — Claude Integration + Guardrails (`src/generate.py`)
- System prompt: medical disclaimer, out-of-scope refusal, no dangerous advice
- Prompt template: retrieved Q&A pairs as context + user question
- `generate_answer(query)` — calls retrieve() then Claude Sonnet

### [DONE] Step 6 — Logging (`src/logger.py`)
- Create `query_logs` table in Supabase: `id`, `query`, `response`, `latency_ms`, `sources`, `timestamp`
- Log every query/response after Step 5

### [DONE] Step 7 — Streamlit Chat UI (`src/app.py`)
- Chat message layout
- Wires: user input → retrieve → Claude → display answer
- Show sources (which Q&A pairs were retrieved) and latency
- Sidebar: basic stats (queries today, avg latency)

### [DONE] Step 8 — Analytics Dashboard
- Query `query_logs` from Supabase
- Visualize: query volume over time, latency distribution, task type breakdown

---

## Key Decisions
- Embed **questions only** for retrieval speed; pass **full note** to Claude for answer quality
- Supabase replaces both ChromaDB (vectors) and SQLite (logs) — single DB, cloud from day 1
- Start with 2000-row sample, scale to full 158k after pipeline is validated
