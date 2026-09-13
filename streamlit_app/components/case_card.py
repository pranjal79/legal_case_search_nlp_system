import streamlit as st

def render_case_card(case: dict, show_score: bool = True):
    with st.container(border=True):
        st.subheader(case.get("case_title", "Untitled Case"))
        cols = st.columns(4)
        cols[0].markdown(f"**Year:** {case.get('year', 'N/A')}")
        cols[1].markdown(f"**Outcome:** {case.get('outcome', 'unknown')}")
        cols[2].markdown(f"**Court:** {case.get('court', 'N/A')}")
        if show_score and "similarity_score" in case:
            cols[3].markdown(f"**Similarity:** {case['similarity_score']:.3f}")

        st.markdown(f"**Facts:** {case.get('case_facts', '')[:300]}...")

        if case.get("legal_keywords"):
            st.caption(f"Keywords: {case['legal_keywords']}")

        if st.button("View Details", key=f"view_{case.get('case_id')}"):
            st.session_state["selected_case_id"] = case.get("case_id")
            st.switch_page("pages/2_Case_Details.py")