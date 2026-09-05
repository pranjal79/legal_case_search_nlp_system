import pandas as pd
import json
from pathlib import Path
from collections import defaultdict

INPUT_PATH = Path("data/processed/cases_clean.parquet")
OUTPUT_PATH = Path("models/judge_stats.json")

def analyze_judges():
    df = pd.read_parquet(INPUT_PATH, columns=["judges", "outcome"])
    df = df[df["judges"].notna() & (df["judges"] != "")]

    stats = defaultdict(lambda: {"total_cases": 0, "allowed": 0, "dismissed": 0, "other": 0})

    for _, row in df.iterrows():
        judge_names = [j.strip() for j in str(row["judges"]).split("; ") if j.strip()]
        outcome = row["outcome"]
        for judge in judge_names:
            stats[judge]["total_cases"] += 1
            if outcome == "allowed":
                stats[judge]["allowed"] += 1
            elif outcome == "dismissed":
                stats[judge]["dismissed"] += 1
            else:
                stats[judge]["other"] += 1

    filtered = {k: v for k, v in stats.items() if v["total_cases"] >= 5}
    for judge, s in filtered.items():
        s["allowed_rate"] = round(s["allowed"] / s["total_cases"], 3)
        s["dismissed_rate"] = round(s["dismissed"] / s["total_cases"], 3)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(filtered, f, indent=2)

    print(f"Analyzed {len(filtered)} judges (>=5 cases each)")
    print(f"Saved to {OUTPUT_PATH}")

    top_5 = sorted(filtered.items(), key=lambda x: x[1]["total_cases"], reverse=True)[:5]
    print("\nMost active judges in dataset:")
    for judge, s in top_5:
        print(f"  {judge}: {s['total_cases']} cases, allowed_rate={s['allowed_rate']}")

if __name__ == "__main__":
    analyze_judges()