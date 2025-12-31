import streamlit as st


import streamlit as st

# ------------------------------------------------
# Import your RAG pipeline
# ------------------------------------------------
# IMPORTANT:
# We are NOT re-writing logic.
# We are importing the already-built pipeline.

from rag_pipeline import get_rag_app

rag_app = get_rag_app()
   # see NOTE below


# ------------------------------------------------
# Streamlit UI
# ------------------------------------------------

st.set_page_config(page_title="Insurance Policy RAG", layout="wide")

st.title("🧠 Agentic RAG – Insurance Policy Assistant")
st.markdown(
    """
Ask questions about the insurance policy.
Answers are generated **strictly from the policy text**.
"""
)

query = st.text_input(
    "Enter your question:",
    placeholder="e.g. What is the difference between Optima Secure and Optima Super Secure?"
)

if st.button("Get Answer"):
    if not query.strip():
        st.warning("Please enter a valid question.")
    else:
        with st.spinner("Analyzing policy..."):
            result = rag_app.invoke({"query": query})
            st.success("Answer:")
            st.write(result["answer"])
