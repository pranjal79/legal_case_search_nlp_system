import streamlit as st

def render_sidebar():
    st.sidebar.header("Filters")
    year = st.sidebar.number_input("Year (0 = any)", min_value=0, max_value=2023, value=0)
    outcome = st.sidebar.selectbox(
        "Outcome",
        ["Any", "allowed", "dismissed", "partly_allowed", "remanded", "unknown"],
    )
    top_k = st.sidebar.slider("Number of results", min_value=5, max_value=50, value=10)

    return {
        "year": year if year > 0 else None,
        "outcome": None if outcome == "Any" else outcome,
        "top_k": top_k,
    }