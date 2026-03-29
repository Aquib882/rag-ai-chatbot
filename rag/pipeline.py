from rag.chunking import chunk_text
from rag.embedding import get_embeddings
from rag.vector_store import create_index
from rag.retriever import retrieve
from rag.generator import generate_answer
from rag.config import CHUNK_SIZE, CHUNK_OVERLAP

def init_pipeline(text):
    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    embeddings = get_embeddings(chunks)
    index = create_index(embeddings)

    return index, chunks

def run_query(query, index, chunks):
    retrieved_chunks = retrieve(query, index, chunks)
    answer = generate_answer(query, retrieved_chunks)

    return answer, retrieved_chunks