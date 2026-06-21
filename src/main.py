import streamlit as st
from query import query_rag_pipeline
from ingest import ingest_documents

st.set_page_config(page_title="Document Q&A Bot", layout="wide")

st.title("📚 Document Q&A Bot with RAG")

st.sidebar.header("Setup")
if st.sidebar.button("Ingest Documents"):
    with st.spinner("Ingesting documents..."):
        ingest_documents()
    st.success("Documents indexed successfully!")

user_query = st.text_input("Ask a question about your documents:")

if st.button("Get Answer") and user_query:
    with st.spinner("Thinking..."):
        result = query_rag_pipeline(user_query)
    
    st.subheader("Answer")
    st.write(result["answer"])
    
    st.subheader("Sources")
    for cit in result["citations"]:
        st.write(f"- {cit}")