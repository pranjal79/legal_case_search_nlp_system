from src.search.semantic_search import LegalCaseSearchEngine
from src.utils.mlflow_logger import mlflow_run
import mlflow

# Each test query paired with a keyword that SHOULD appear in a relevant result
TEST_QUERIES = [
    ("right to privacy under article 21", "privacy"),
    ("anticipatory bail in criminal case", "bail"),
    ("land acquisition compensation dispute", "land acquisition"),
    ("breach of contract damages", "contract"),
    ("habeas corpus fundamental rights", "habeas corpus"),
]

def evaluate_search_quality(engine: LegalCaseSearchEngine, top_k: int = 5):
    hits = 0
    total = len(TEST_QUERIES)
    details = []

    for query, expected_keyword in TEST_QUERIES:
        results = engine.search(query, top_k=top_k)
        found = any(
            expected_keyword.lower() in (r.get("case_title", "") + r.get("legal_keywords", "")).lower()
            for r in results
        )
        hits += int(found)
        details.append((query, found, len(results)))

    precision_at_k = hits / total
    return precision_at_k, details

def run_evaluation():
    engine = LegalCaseSearchEngine()

    with mlflow_run(run_name="faiss_search_eval"):
        mlflow.log_param("model_name", "all-MiniLM-L6-v2")
        mlflow.log_param("index_type", "IndexFlatIP")
        mlflow.log_param("top_k", 5)

        precision, details = evaluate_search_quality(engine, top_k=5)
        mlflow.log_metric("precision_at_5", precision)

        for query, found, n_results in details:
            print(f"  [{'HIT' if found else 'MISS'}] '{query}' -> {n_results} results")

        print(f"\nPrecision@5: {precision:.2f}")

    engine.close()

if __name__ == "__main__":
    run_evaluation()