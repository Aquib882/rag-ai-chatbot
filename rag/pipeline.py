from chunking import chunk_text
from embedding import get_embeddings
from vector_store import create_index
from retriever import retrieve
from generator import generate_answer
from config import CHUNK_SIZE, CHUNK_OVERLAP

def init_pipeline(text):
    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    embeddings = get_embeddings(chunks)
    index = create_index(embeddings)

    return index, chunks

def run_query(query, index, chunks):
    retrieved_chunks = retrieve(query, index, chunks)
    answer = generate_answer(query, retrieved_chunks)

    return answer, retrieved_chunks