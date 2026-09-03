import fitz  # PyMuPDF
import zipfile
import pandas as pd
from pathlib import Path
import re

RAW_DATA_DIR = Path("data/raw/supreme_court_pdfs")
OUTPUT_PATH = Path("data/processed/cases_raw.csv")

def parse_case_title(filename: str):
    """A_K_Gopalan_vs_State_of_Madras.PDF -> title, petitioner, respondent"""
    name = filename.rsplit(".", 1)[0]
    title = name.replace("_", " ")
    if " vs " in title.lower():
        parts = re.split(r"\s+vs\s+", title, flags=re.IGNORECASE)
        petitioner = parts[0].strip()
        respondent = parts[1].strip() if len(parts) > 1 else ""
    else:
        petitioner, respondent = title, ""
    return title, petitioner, respondent

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Open a PDF directly from in-memory bytes (no disk extraction needed)."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def process_zip(zip_path: Path, year: int, case_id_start: int):
    records = []
    case_id = case_id_start
    with zipfile.ZipFile(zip_path, "r") as zf:
        pdf_names = [n for n in zf.namelist() if n.lower().endswith(".pdf")]
        for pdf_name in pdf_names:
            try:
                pdf_bytes = zf.read(pdf_name)
                judgment_text = extract_text_from_pdf_bytes(pdf_bytes)
                filename = Path(pdf_name).name
                title, petitioner, respondent = parse_case_title(filename)

                records.append({
                    "case_id": f"case_{case_id:06d}",
                    "case_title": title,
                    "year": year,
                    "court": "Supreme Court of India",
                    "petitioner": petitioner,
                    "respondent": respondent,
                    "judgment_text": judgment_text,
                    "source_file": f"{zip_path.name}::{pdf_name}",
                    "text_length": len(judgment_text),
                })
                case_id += 1
            except Exception as e:
                print(f"  [WARN] Failed to process {pdf_name}: {e}")
    return records, case_id

def run_extraction():
    all_records = []
    case_id_counter = 1

    zip_files = sorted(RAW_DATA_DIR.glob("*.zip"))
    print(f"Found {len(zip_files)} year-zip files.")

    for zip_path in zip_files:
        year = int(zip_path.stem)  # "1950.zip" -> 1950
        print(f"Processing {zip_path.name}...")
        records, case_id_counter = process_zip(zip_path, year, case_id_counter)
        all_records.extend(records)
        print(f"  -> {len(records)} cases extracted")

    df = pd.DataFrame(all_records)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(df)} total cases to {OUTPUT_PATH}")
    return df

if __name__ == "__main__":
    run_extraction()