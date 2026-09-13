import streamlit as st

st.set_page_config(
    page_title="Legal Case Search & Precedent Finder",
    page_icon="⚖️",
    layout="wide",
)

st.title("⚖️ Legal Case Law Search & Precedent Finder")
st.markdown("""
Search 26,000+ Supreme Court of India judgments (1950–2023) using natural language.
Use the sidebar to navigate between search, case details, citation graphs, and judge analytics.
""")

st.info("👈 Select a page from the sidebar to get started.")