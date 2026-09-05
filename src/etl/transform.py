import pandas as pd
import re
from pathlib import Path

INPUT_PATH = Path("data/processed/cases_raw.csv")
OUTPUT_PATH = Path("data/processed/cases_clean.csv")

LEGAL_KEYWORDS = [
    "constitutional", "fundamental rights", "writ petition", "habeas corpus",
    "article 21", "article 14", "article 19", "due process", "natural justice",
    "bail", "anticipatory bail", "criminal appeal", "civil appeal",
    "negligence", "contract", "breach", "damages", "compensation",
    "arbitration", "injunction", "specific performance", "tort",
    "land acquisition", "eminent domain", "taxation", "income tax",
    "labour law", "industrial dispute", "matrimonial", "divorce",
    "custody", "inheritance", "succession", "property rights",
]

OUTCOME_PATTERNS = {
    "allowed": r"\b(appeal|petition)\s+is\s+allowed\b",
    "dismissed": r"\b(appeal|petition)\s+is\s+dismissed\b",
    "partly_allowed": r"\bpartly\s+allowed\b",
    "remanded": r"\bremanded\b|\bremand(ed)?\s+to\b",
}

CITATION_PATTERNS = [
    r"AIR\s+\d{4}\s+SC\s+\d+",           # AIR 1973 SC 1461
    r"\(\d{4}\)\s+\d+\s+SCC\s+\d+",       # (1973) 4 SCC 225
]

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\n\s*\d+\s*\n", "\n", text)      # standalone page numbers
    text = re.sub(r"[ \t]+", " ", text)               # collapse whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)             # collapse blank lines
    return text.strip()

def detect_outcome(text: str) -> str:
    tail = text[-1000:].lower() if len(text) > 1000 else text.lower()
    for outcome, pattern in OUTCOME_PATTERNS.items():
        if re.search(pattern, tail, re.IGNORECASE):
            return outcome
    return "unknown"

def extract_citations(text: str) -> str:
    found = []
    for pattern in CITATION_PATTERNS:
        found.extend(re.findall(pattern, text))
    return "; ".join(sorted(set(found)))

def extract_legal_keywords(text: str) -> str:
    text_lower = text.lower()
    found = [kw for kw in LEGAL_KEYWORDS if kw in text_lower]
    return "; ".join(found)

JOURNAL_ABBREVIATIONS = {"all", "cri", "air", "scc", "scr", "mad", "bom", "cal", "del"}

def extract_judges(text: str) -> str:
    head = text[:500]
    judge_lines = re.findall(r"([A-Z][A-Za-z\.\s]{2,40}?,?\s*J\.)", head)
    filtered = []
    for j in judge_lines:
        first_word = j.strip().split(".")[0].strip().lower()
        if first_word not in JOURNAL_ABBREVIATIONS and len(j.strip()) > 5:
            filtered.append(j.strip())
    return "; ".join(set(filtered))

def run_transform():
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} raw cases.")

    df["judgment_text_clean"] = df["judgment_text"].apply(clean_text)
    df["case_facts"] = df["judgment_text_clean"].str[:500]
    df["text_preview"] = df["judgment_text_clean"].str[:600]
    df["outcome"] = df["judgment_text_clean"].apply(detect_outcome)
    df["citations"] = df["judgment_text_clean"].apply(extract_citations)
    df["legal_keywords"] = df["judgment_text_clean"].apply(extract_legal_keywords)
    df["judges"] = df["judgment_text_clean"].apply(extract_judges)

    final_cols = [
        "case_id", "case_title", "year", "court", "petitioner", "respondent",
        "judges", "case_facts", "judgment_text_clean", "text_preview",
        "outcome", "citations", "legal_keywords", "source_file", "text_length",
    ]
    df_final = df[final_cols]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df_final)} cleaned cases to {OUTPUT_PATH}")

    print("\nOutcome distribution:")
    print(df_final["outcome"].value_counts())

    return df_final

if __name__ == "__main__":
    run_transform()