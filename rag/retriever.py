from rag.embedding import model
from rag.vector_store import search_index
from rag.config import TOP_K

def retrieve(query, index, chunks):
    query_embedding = model.encode([query])[0]
    indices = search_index(query_embedding, index, TOP_K)
    return [chunks[i] for i in indices]