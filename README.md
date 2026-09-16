# Legal Case Law Search & Precedent Finder

A natural-language search system for 26,000+ Supreme Court of India judgments (1950–2023), with semantic case search, automatic summarization, outcome prediction, citation network analysis, and judge analytics.

**Live App:** [https://legalcasesearchnlpsystem-ibjb4ycd8xup6qyd42et4e.streamlit.app](https://legalcasesearchnlpsystem-ibjb4ycd8xup6qyd42et4e.streamlit.app)

---

## Overview

Given a plain-English query like *"right to privacy under article 21"* or *"anticipatory bail in criminal case"*, this system returns the most relevant Supreme Court judgments using semantic (embedding-based) search rather than plain keyword matching. From there, users can:

- View full case details and auto-generated summaries
- Get a predicted case outcome with confidence scores
- Explore the citation network to find influential precedents
- Browse judge-level statistics (case counts, allowed/dismissed rates)

---

## Features

| Feature | Description |
|---|---|
| 🔍 **Semantic Search** | FAISS + `all-MiniLM-L6-v2` sentence embeddings for natural-language case search, with year/outcome filters |
| 📄 **Case Summarization** | Abstractive summaries generated with `sshleifer/distilbart-cnn-12-6` (BART) |
| 🔮 **Outcome Prediction** | TF-IDF + Random Forest classifier predicting case outcome (allowed / dismissed / partly allowed / remanded) |
| 🕸️ **Citation Graph** | NetworkX-based citation network with PageRank to surface the most influential precedents |
| 👨‍⚖️ **Judge Analytics** | Per-judge case counts and outcome-rate breakdowns |

---

## Tech Stack

- **Language:** Python 3.11
- **PDF Extraction:** PyMuPDF (fitz)
- **Data Processing:** pandas, DuckDB, Apache Parquet
- **Embeddings & Search:** sentence-transformers, FAISS (`IndexFlatIP`, cosine similarity)
- **Summarization:** HuggingFace Transformers (BART, loaded directly via `AutoModelForSeq2SeqLM`)
- **Classification:** scikit-learn (TF-IDF + Logistic Regression / Random Forest / Gradient Boosting, compared via MLflow)
- **Graph Analysis:** NetworkX (citation graph + PageRank)
- **Experiment Tracking:** MLflow (hosted on DAGsHub)
- **Data Versioning:** DVC + DAGsHub (raw PDF dataset only — derived artifacts are pipeline outputs)
- **Frontend:** Streamlit + Plotly
- **Deployment:** Streamlit Community Cloud

---

## Architecture Notes

- **Storage:** DuckDB + Parquet is used for both local development and deployment — no database server required. (An earlier MongoDB-based design was dropped in favor of this simpler, serverless approach.)
- **Deployment dataset:** The full processed dataset (~425MB, including raw judgment text) is used for local development. A slimmed-down version (`cases_deploy.parquet`, ~15MB, with the heaviest text column removed) is committed directly to the repo and used in the deployed app, since Streamlit Cloud has no access to the DVC remote at runtime.
- **Pipeline reproducibility:** Only the raw PDF dataset (`data/raw/supreme_court_pdfs/`) is tracked via DVC and pushed to DAGsHub. All derived artifacts (cleaned CSVs, Parquet files, FAISS index, trained models) are defined as `dvc.yaml` pipeline stages and can be regenerated locally with `dvc repro`, rather than being pushed as large binaries.

---

## Project Structure

```
legal-case-search/
├── data/
│   ├── raw/supreme_court_pdfs/    # DVC-tracked, zipped by year
│   ├── processed/                 # cases_raw.csv, cases_clean.csv, .parquet
│   └── embeddings/                # FAISS index + case_ids.pkl
├── src/
│   ├── etl/                       # extract, transform, export_parquet
│   ├── ml/                        # embeddings, classifier, summarizer, citation graph, judge analysis
│   ├── search/                    # semantic search engine, DuckDB client
│   └── utils/                     # MLflow logger, config loader
├── streamlit_app/
│   ├── app.py                     # entry point
│   ├── pages/                     # Search, Case Details, Citation Graph, Judge Analytics
│   └── components/                # sidebar, case card
├── models/                        # trained classifier, vectorizer, citation graph, judge stats
├── configs/, params.yaml, dvc.yaml # pipeline configuration
└── requirements.txt
```

---

## Running Locally

```bash
git clone https://github.com/pranjal79/legal_case_search_nlp_system.git
cd legal_case_search_nlp_system
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Pull raw data from DVC remote, then regenerate all derived artifacts
dvc pull
dvc repro

# Run the app
streamlit run streamlit_app/app.py
```

---

## Known Limitations

- **Outcome classifier accuracy:** The best model (Random Forest) achieves an F1-macro of ~0.40. This reflects a genuinely imbalanced dataset (majority of labeled cases are "allowed") and the limited signal available in the 500-character case-facts snippet used as input. Predictions should be treated as indicative, not authoritative.
- **Outcome labeling coverage:** Roughly 79% of cases have an outcome of "unknown," since the regex-based outcome detector only catches a few common phrasings (e.g., "the appeal is allowed"). Many outcome statements in older or differently-worded judgments aren't captured.
- **Judge name extraction:** Judge names are extracted via a regex heuristic scanning the first 500 characters of each judgment. This is noisy — legal citation abbreviations were filtered out, but some misidentifications may remain.
- **Citation graph coverage:** Citations are only detected in `AIR YYYY SC NNNN` and `(YYYY) N SCC NNNN` formats. Citations to cases outside these formats, or to cases not present in the 26,688-document corpus, are not captured — so the graph reflects internal, in-corpus citation patterns rather than each case's full citation history.
- **Case summaries:** Summaries are generated from the first ~500 characters of judgment text (not the full document), so they reflect only the opening portion of each case.

---

## Deployment

Deployed on **Streamlit Community Cloud**, building directly from `requirements.txt` (no Docker involved in the deployment path). Secrets (DAGsHub credentials, MLflow tracking URI) are configured via Streamlit Cloud's Secrets manager.

---

## Dataset

Supreme Court of India judgments, 1950–2023, sourced as individual PDF files and organized by year. Full text extracted via PyMuPDF; structured fields (petitioner, respondent, outcome, citations, legal keywords, judges) derived via a custom ETL pipeline.
