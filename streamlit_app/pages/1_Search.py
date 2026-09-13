import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.search.semantic_search import LegalCaseSearchEngine
from streamlit_app.components.sidebar import render_sidebar
from streamlit_app.components.case_card import render_case_card

st.set_page_config(page_title="Search", layout="wide")
st.title("🔍 Semantic Case Search")

@st.cache_resource
def get_engine():
    return LegalCaseSearchEngine()

engine = get_engine()
filters = render_sidebar()

query = st.text_input(
    "Search cases in natural language",
    placeholder="e.g. right to privacy under article 21",
)

example_queries = [
    "anticipatory bail in criminal case",
    "land acquisition compensation dispute",
    "breach of contract damages",
]
st.caption("Try: " + " | ".join(f"`{q}`" for q in example_queries))

if query:
    with st.spinner("Searching..."):
        results = engine.search(
            query,
            top_k=filters["top_k"],
            year=filters["year"],
            outcome=filters["outcome"],
        )

    if not results:
        st.warning("No matching cases found. Try adjusting your filters or query.")
    else:
        st.success(f"Found {len(results)} results")
        for case in results:
            render_case_card(case)