import pandas as pd
from pathlib import Path

INPUT_PATH = Path("data/processed/cases_clean.parquet")
OUTPUT_PATH = Path("data/processed/cases_deploy.parquet")

# Columns actually used by the Streamlit app + search engine.
# judgment_text_clean is NOT included — it's the largest column and unused at query time.
DEPLOY_COLUMNS = [
    "case_id", "case_title", "year", "court", "petitioner", "respondent",
    "judges", "case_facts", "text_preview", "outcome", "citations",
    "legal_keywords", "text_length",
]

def export_deploy_parquet():
    df = pd.read_parquet(INPUT_PATH, columns=DEPLOY_COLUMNS)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, engine="pyarrow", compression="snappy", index=False)

    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"Saved deployment Parquet: {OUTPUT_PATH} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    export_deploy_parquet()