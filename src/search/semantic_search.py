import faiss
import pickle
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from src.search.duckdb_client import DuckDBClient

INDEX_PATH = Path("data/embeddings/faiss_index.bin")
IDS_PATH = Path("data/embeddings/case_ids.pkl")
MODEL_NAME = "all-MiniLM-L6-v2"

class LegalCaseSearchEngine:
    def __init__(self):
        self.index = faiss.read_index(str(INDEX_PATH))
        with open(IDS_PATH, "rb") as f:
            self.case_ids = pickle.load(f)
        self.model = SentenceTransformer(MODEL_NAME)
        self.db = DuckDBClient()

    def search(self, query: str, top_k: int = 10, year: int = None, outcome: str = None):
        query_vec = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_vec)

        # Over-fetch when filters are applied, since FAISS doesn't filter natively
        fetch_k = top_k * 5 if (year or outcome) else top_k
        scores, indices = self.index.search(query_vec.astype(np.float32), fetch_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            case_id = self.case_ids[idx]
            case = self.db.get_case_by_id(case_id)
            if case is None:
                continue
            if year and case.get("year") != year:
                continue
            if outcome and case.get("outcome") != outcome:
                continue
            case["similarity_score"] = float(score)
            results.append(case)
            if len(results) >= top_k:
                break

        return results

    def get_case_by_id(self, case_id: str):
        return self.db.get_case_by_id(case_id)

    def get_similar_to_case(self, case_id: str, top_k: int = 5):
        idx = self.case_ids.index(case_id)
        case_vec = self.index.reconstruct(idx).reshape(1, -1)
        scores, indices = self.index.search(case_vec, top_k + 1)  # +1 to skip itself

        results = []
        for score, i in zip(scores[0], indices[0]):
            similar_id = self.case_ids[i]
            if similar_id == case_id:
                continue
            case = self.db.get_case_by_id(similar_id)
            if case:
                case["similarity_score"] = float(score)
                results.append(case)
        return results[:top_k]

    def close(self):
        self.db.close()