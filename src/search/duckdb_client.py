import duckdb
from pathlib import Path

# Anchor to the project root (two levels up from this file: src/search/ -> project root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEPLOY_PARQUET_PATH = PROJECT_ROOT / "data" / "processed" / "cases_deploy.parquet"
FULL_PARQUET_PATH = PROJECT_ROOT / "data" / "processed" / "cases_clean.parquet"

PARQUET_PATH = FULL_PARQUET_PATH if FULL_PARQUET_PATH.exists() else DEPLOY_PARQUET_PATH

class DuckDBClient:
    def __init__(self, parquet_path: Path = PARQUET_PATH):
        if not parquet_path.exists():
            raise FileNotFoundError(
                f"No parquet file found at {parquet_path.resolve()}. "
                f"Checked FULL={FULL_PARQUET_PATH.resolve()} and DEPLOY={DEPLOY_PARQUET_PATH.resolve()}."
            )
        self.con = duckdb.connect(database=":memory:")
        self.con.execute(f"""
            CREATE TABLE cases AS
            SELECT * FROM read_parquet('{parquet_path}')
        """)

    def get_case_by_id(self, case_id: str):
        result = self.con.execute(
            "SELECT * FROM cases WHERE case_id = ?", [case_id]
        ).fetchdf()
        return result.iloc[0].to_dict() if not result.empty else None

    def filter_cases(self, year: int = None, outcome: str = None, limit: int = 50):
        query = "SELECT * FROM cases WHERE 1=1"
        params = []
        if year:
            query += " AND year = ?"
            params.append(year)
        if outcome:
            query += " AND outcome = ?"
            params.append(outcome)
        query += " LIMIT ?"
        params.append(limit)
        return self.con.execute(query, params).fetchdf()

    def get_stats(self):
        return self.con.execute("""
            SELECT year, outcome, COUNT(*) as count
            FROM cases GROUP BY year, outcome ORDER BY year
        """).fetchdf()

    def close(self):
        self.con.close()