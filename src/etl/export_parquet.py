import pandas as pd
from pathlib import Path

INPUT_PATH = Path("data/processed/cases_clean.csv")
OUTPUT_PATH = Path("data/processed/cases_clean.parquet")

def export_parquet():
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} cleaned cases from CSV.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, engine="pyarrow", compression="snappy", index=False)

    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"Saved Parquet file: {OUTPUT_PATH} ({size_mb:.2f} MB)")

    return df

if __name__ == "__main__":
    export_parquet()