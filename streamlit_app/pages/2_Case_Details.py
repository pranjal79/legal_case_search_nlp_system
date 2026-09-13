import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.search.semantic_search import LegalCaseSearchEngine
from src.ml.summarizer import CaseSummarizer
from src.ml.predictor import OutcomePredictor

st.set_page_config(page_title="Case Detail", layout="wide")
st.title("📄 Case Detail")

@st.cache_resource
def get_engine():
    return LegalCaseSearchEngine()

@st.cache_resource
def get_summarizer():
    return CaseSummarizer()

@st.cache_resource
def get_predictor():
    return OutcomePredictor()

case_id = st.session_state.get("selected_case_id")

if not case_id:
    st.warning("No case selected. Go to the Search page and click 'View Details' on a case.")
    st.stop()

engine = get_engine()
case = engine.get_case_by_id(case_id)

if not case:
    st.error(f"Case {case_id} not found.")
    st.stop()

st.header(case["case_title"])
st.caption(f"{case['court']} — {case['year']} — Outcome: {case['outcome']}")

tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Facts", "Outcome Prediction", "Similar Cases"])

with tab1:
    if st.button("Generate Summary"):
        with st.spinner("Summarizing..."):
            summary = get_summarizer().summarize(case.get("case_facts", ""))
        st.write(summary)

with tab2:
    st.write(case.get("case_facts", "No facts available."))
    if case.get("citations"):
        st.markdown(f"**Citations:** {case['citations']}")
    if case.get("legal_keywords"):
        st.markdown(f"**Legal Keywords:** {case['legal_keywords']}")

with tab3:
    if st.button("Predict Outcome"):
        result = get_predictor().predict(case.get("case_facts", ""), case.get("legal_keywords", ""))
        st.metric("Predicted Outcome", result["predicted_outcome"])
        st.bar_chart(result["confidence_scores"])

with tab4:
    with st.spinner("Finding similar cases..."):
        similar = engine.get_similar_to_case(case_id, top_k=5)
    for sim_case in similar:
        st.markdown(f"- **{sim_case['case_title']}** ({sim_case['year']}) — similarity: {sim_case['similarity_score']:.3f}")