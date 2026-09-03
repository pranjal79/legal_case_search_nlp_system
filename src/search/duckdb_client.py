import duckdb
from pathlib import Path

PARQUET_PATH = Path("data/processed/cases_clean.parquet")

class DuckDBClient:
    def __init__(self, parquet_path: Path = PARQUET_PATH):
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