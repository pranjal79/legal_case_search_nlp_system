import streamlit as st
import pandas as pd
import json
from pathlib import Path

st.set_page_config(page_title="Judge Analytics", layout="wide")
st.title("👨‍⚖️ Judge Analytics")

STATS_PATH = Path("models/judge_stats.json")

if not STATS_PATH.exists():
    st.error("Judge stats not found. Run `python -m src.ml.judge_analysis` first.")
    st.stop()

with open(STATS_PATH) as f:
    stats = json.load(f)

df = pd.DataFrame.from_dict(stats, orient="index")
df.index.name = "judge"
df = df.reset_index().sort_values("total_cases", ascending=False)

st.subheader("Most Active Judges")
st.dataframe(df.head(20), use_container_width=True)

st.subheader("Allowed Rate Distribution")
st.bar_chart(df.set_index("judge")["allowed_rate"].head(20))

judge = st.selectbox("Select a judge for details", df["judge"].tolist())
judge_row = df[df["judge"] == judge].iloc[0]
col1, col2, col3 = st.columns(3)
col1.metric("Total Cases", int(judge_row["total_cases"]))
col2.metric("Allowed Rate", f"{judge_row['allowed_rate']:.1%}")
col3.metric("Dismissed Rate", f"{judge_row['dismissed_rate']:.1%}")