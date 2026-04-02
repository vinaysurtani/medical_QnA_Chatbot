"""
Step 6: Query logging.
Logs every query, response, latency, and sources to Supabase query_logs table.
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client

load_dotenv(Path(__file__).parent.parent / ".env")

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = create_client(
            os.environ["SUPABASE_URL"],
            os.environ["SUPABASE_SERVICE_KEY"],
        )
    return _client


def log_query(query: str, response: str, latency_ms: int, sources: list[dict]):
    """
    Insert one row into query_logs table.
    sources is stored as JSONB — list of {question, task, similarity} dicts.
    """
    try:
        _get_client().table("query_logs").insert({
            "query":      query,
            "response":   response,
            "latency_ms": latency_ms,
            "sources":    json.dumps(sources),
        }).execute()
    except Exception as e:
        # Logging should never crash the main app
        print(f"[logger] Warning: failed to log query — {e}")


if __name__ == "__main__":
    # Smoke test
    log_query(
        query="What medications was the patient discharged with?",
        response="Patient was discharged with sacubitril, valsartan...",
        latency_ms=4200,
        sources=[{"question": "What medications were prescribed?", "task": "Question Answering", "similarity": 0.862}],
    )
    print("Log entry written. Check Supabase → Table Editor → query_logs.")
