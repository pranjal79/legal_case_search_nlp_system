import pandas as pd
import networkx as nx
import pickle
from pathlib import Path

INPUT_PATH = Path("data/processed/cases_clean.parquet")
GRAPH_PATH = Path("models/citation_graph.pkl")

class CitationGraph:
    def __init__(self, build_new: bool = False):
        if build_new or not GRAPH_PATH.exists():
            self.graph = self._build()
        else:
            with open(GRAPH_PATH, "rb") as f:
                self.graph = pickle.load(f)

    def _build(self) -> nx.DiGraph:
        df = pd.read_parquet(INPUT_PATH, columns=["case_id", "case_title", "citations"])
        G = nx.DiGraph()

        for _, row in df.iterrows():
            G.add_node(row["case_id"], title=row["case_title"])

        citation_lookup = {}
        for _, row in df.iterrows():
            if row["citations"]:
                for cite in str(row["citations"]).split("; "):
                    citation_lookup.setdefault(cite.strip(), []).append(row["case_id"])

        for _, row in df.iterrows():
            if not row["citations"]:
                continue
            for cite in str(row["citations"]).split("; "):
                cite = cite.strip()
                if cite in citation_lookup:
                    for target_id in citation_lookup[cite]:
                        if target_id != row["case_id"]:
                            G.add_edge(row["case_id"], target_id, citation=cite)

        print(f"Built graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(GRAPH_PATH, "wb") as f:
            pickle.dump(G, f)

        return G

    def get_pagerank(self, top_n: int = 20):
        scores = nx.pagerank(self.graph)
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [
            {"case_id": cid, "title": self.graph.nodes[cid].get("title", ""), "pagerank": score}
            for cid, score in sorted_scores
        ]

    def get_subgraph_for_viz(self, case_id: str, depth: int = 1):
        if case_id not in self.graph:
            return nx.DiGraph()
        nodes = {case_id}
        frontier = {case_id}
        for _ in range(depth):
            next_frontier = set()
            for n in frontier:
                next_frontier.update(self.graph.predecessors(n))
                next_frontier.update(self.graph.successors(n))
            nodes.update(next_frontier)
            frontier = next_frontier
        return self.graph.subgraph(nodes)

if __name__ == "__main__":
    cg = CitationGraph(build_new=True)
    print("\nTop 10 most-cited (influential) cases by PageRank:")
    for entry in cg.get_pagerank(top_n=10):
        print(f"  {entry['title']} — {entry['pagerank']:.5f}")