import streamlit as st
import plotly.graph_objects as go
import networkx as nx
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.ml.citation_graph import CitationGraph

st.set_page_config(page_title="Citation Graph", layout="wide")
st.title("🕸️ Citation Network")

@st.cache_resource
def get_graph():
    return CitationGraph(build_new=False)

cg = get_graph()

st.subheader("Top 20 Most Influential Cases (PageRank)")
top_cases = cg.get_pagerank(top_n=20)
st.table(top_cases)

st.subheader("Explore a Case's Citation Neighborhood")
case_id = st.text_input("Enter a case_id (e.g. case_000123)")
max_nodes = st.slider("Max nodes to display", min_value=10, max_value=150, value=40)

if case_id:
    subgraph = cg.get_subgraph_for_viz(case_id, depth=1)

    if subgraph.number_of_nodes() == 0:
        st.warning("Case not found or has no citation links.")
    else:
        # Cap node count — if too large, keep only the highest-degree nodes
        # (i.e. the most connected/relevant ones) plus the queried case itself
        if subgraph.number_of_nodes() > max_nodes:
            st.info(
                f"This case's neighborhood has {subgraph.number_of_nodes()} nodes — "
                f"showing the top {max_nodes} most connected for readability."
            )
            degrees = dict(subgraph.degree())
            top_node_ids = sorted(degrees, key=degrees.get, reverse=True)[:max_nodes]
            if case_id not in top_node_ids:
                top_node_ids.append(case_id)
            subgraph = subgraph.subgraph(top_node_ids)

        pos = nx.spring_layout(subgraph, seed=42, k=0.8)  # k spreads nodes further apart

        edge_x, edge_y = [], []
        for edge in subgraph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        node_x = [pos[n][0] for n in subgraph.nodes()]
        node_y = [pos[n][1] for n in subgraph.nodes()]
        node_hover_text = [subgraph.nodes[n].get("title", n) for n in subgraph.nodes()]
        # Highlight the queried case in a different color
        node_colors = ["crimson" if n == case_id else "steelblue" for n in subgraph.nodes()]
        node_sizes = [16 if n == case_id else 9 for n in subgraph.nodes()]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y, mode="lines",
            line=dict(width=0.5, color="lightgray"), hoverinfo="none",
        ))
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y, mode="markers",
            marker=dict(size=node_sizes, color=node_colors),
            text=node_hover_text, hoverinfo="text",  # labels only show on hover now
        ))
        fig.update_layout(
            showlegend=False, height=650,
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
        )
        st.plotly_chart(fig, use_container_width=True)