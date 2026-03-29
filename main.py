import streamlit as st
from rag.loader import load_text
from rag.pipeline import init_pipeline

st.title("📄 RAG Chatbot")

uploaded_file = st.file_uploader("Upload a text file")

if uploaded_file:
    text = load_text(uploaded_file)
    index, chunks = init_pipeline(text)

    st.success("File processed successfully!")

    query = st.text_input("Ask a question:")

    if query:
        answer, retrieved_chunks = run_query(query, index, chunks)

        st.subheader("Answer")
        st.write(answer)

        with st.expander("Retrieved Context"):
            for c in retrieved_chunks:
                st.write(c)