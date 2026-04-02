"""
Step 8: Analytics dashboard.
Reads query_logs from Supabase and visualises usage, latency, and retrieval patterns.
"""

import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from supabase import create_client

load_dotenv(Path(__file__).parent.parent / ".env")

st.set_page_config(
    page_title="Medical Q&A — Analytics",
    page_icon="📊",
    layout="wide",
)

st.title("Analytics Dashboard")
st.caption("Usage and performance metrics from query_logs.")


@st.cache_data(ttl=60)
def load_logs() -> pd.DataFrame:
    client = create_client(
        os.environ["SUPABASE_URL"],
        os.environ["SUPABASE_SERVICE_KEY"],
    )
    response = client.table("query_logs").select("*").order("created_at").execute()
    rows = response.data
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df["hour"]       = df["created_at"].dt.floor("h")
    df["date"]       = df["created_at"].dt.date

    # Parse sources JSONB → extract task types
    def extract_tasks(sources_val):
        if not sources_val:
            return []
        if isinstance(sources_val, str):
            sources_val = json.loads(sources_val)
        return [s.get("task", "unknown") for s in sources_val]

    df["tasks"]       = df["sources"].apply(extract_tasks)
    df["top_task"]    = df["tasks"].apply(lambda t: t[0] if t else "no results")
    df["num_sources"] = df["tasks"].apply(len)

    return df


df = load_logs()

if df.empty:
    st.warning("No queries logged yet. Ask some questions in the chat app first.")
    st.stop()

# --- Top metrics ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total queries",     len(df))
col2.metric("Avg latency",       f"{int(df['latency_ms'].mean())}ms")
col3.metric("p95 latency",       f"{int(df['latency_ms'].quantile(0.95))}ms")
col4.metric("No-results rate",   f"{(df['num_sources'] == 0).mean() * 100:.1f}%")

st.divider()

# --- Query volume over time ---
st.subheader("Query volume over time")
volume = df.groupby("hour").size().reset_index(name="queries")
st.bar_chart(volume.set_index("hour")["queries"])

st.divider()

# --- Latency distribution ---
st.subheader("Latency distribution")
bins    = [0, 2000, 4000, 6000, 8000, 10000, 99999]
labels  = ["<2s", "2-4s", "4-6s", "6-8s", "8-10s", ">10s"]
df["latency_bucket"] = pd.cut(df["latency_ms"], bins=bins, labels=labels)
lat_dist = df["latency_bucket"].value_counts().reindex(labels).fillna(0)
st.bar_chart(lat_dist)

st.divider()

# --- Task type breakdown ---
st.subheader("Top task type retrieved (per query)")
task_counts = df["top_task"].value_counts()
st.bar_chart(task_counts)

st.divider()

# --- Recent queries table ---
st.subheader("Recent queries")
cols_to_show = ["created_at", "query", "latency_ms", "top_task", "num_sources"]
recent = (
    df[cols_to_show]
    .sort_values("created_at", ascending=False)
    .head(20)
    .rename(columns={
        "created_at":  "Timestamp",
        "query":       "Query",
        "latency_ms":  "Latency (ms)",
        "top_task":    "Top Task",
        "num_sources": "Sources",
    })
)
recent["Timestamp"] = recent["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
st.dataframe(recent, use_container_width=True, hide_index=True)

if st.button("Refresh"):
    st.cache_data.clear()
    st.rerun()
