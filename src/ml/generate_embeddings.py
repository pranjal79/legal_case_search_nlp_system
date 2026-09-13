import pandas as pd
import numpy as np
import faiss
import pickle
import gc
from pathlib import Path
from sentence_transformers import SentenceTransformer

INPUT_PATH = Path("data/processed/cases_clean.parquet")
INDEX_PATH = Path("data/embeddings/faiss_index.bin")
IDS_PATH = Path("data/embeddings/case_ids.pkl")
MODEL_NAME = "all-MiniLM-L6-v2"

# Only load the columns actually needed for embedding text —
# judgment_text_clean is huge and NOT needed here, so we never load it.
COLUMNS_NEEDED = ["case_id", "case_title", "case_facts", "legal_keywords"]

CHUNK_SIZE = 2000  # cases per batch — keeps memory bounded regardless of dataset size


def build_embedding_text(row) -> str:
    """Combine fields into one text blob to embed per case."""
    parts = [
        str(row.get("case_title", "")),
        str(row.get("case_facts", "")),
        str(row.get("legal_keywords", "")),
    ]
    return " ".join(p for p in parts if p and p != "nan")


def generate_embeddings():
    print(f"Loading only columns: {COLUMNS_NEEDED}")
    df = pd.read_parquet(INPUT_PATH, columns=COLUMNS_NEEDED)
    total_cases = len(df)
    print(f"Loaded {total_cases} cases (lightweight columns only).")

    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    dim = model.get_sentence_embedding_dimension()  # 384 for MiniLM-L6-v2
    index = faiss.IndexFlatIP(dim)
    case_ids = []

    num_chunks = (total_cases + CHUNK_SIZE - 1) // CHUNK_SIZE
    print(f"Processing in {num_chunks} chunks of {CHUNK_SIZE} cases each...\n")

    for chunk_idx in range(num_chunks):
        start = chunk_idx * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, total_cases)
        chunk_df = df.iloc[start:end]

        texts = chunk_df.apply(build_embedding_text, axis=1).tolist()
        chunk_ids = chunk_df["case_id"].tolist()

        embeddings = model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        faiss.normalize_L2(embeddings)
        index.add(embeddings.astype(np.float32))
        case_ids.extend(chunk_ids)

        print(f"  Chunk {chunk_idx + 1}/{num_chunks} done "
              f"({end}/{total_cases} cases, index size: {index.ntotal})")

        # Free memory between chunks
        del texts, embeddings, chunk_df
        gc.collect()

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))

    with open(IDS_PATH, "wb") as f:
        pickle.dump(case_ids, f)

    print(f"\nSaved FAISS index: {INDEX_PATH} ({index.ntotal} vectors, dim={dim})")
    print(f"Saved case_ids: {IDS_PATH}")


if __name__ == "__main__":
    generate_embeddings()