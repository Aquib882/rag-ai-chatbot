import numpy as np

def create_index(embeddings):
    return np.array(embeddings)

def search_index(query_embedding, index, top_k):
    scores = np.dot(index, query_embedding)
    top_indices = scores.argsort()[-top_k:][::-1]
    return top_indices