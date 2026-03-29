from groq import Groq
from config import API_KEY, MODEL_NAME

# Initialize client
client = Groq(api_key=API_KEY)

def generate_answer(query, context_chunks):
    """
    query: user question
    context_chunks: top retrieved chunks from vector search
    """

    # Combine chunks into one context
    context = "\n\n".join(context_chunks)

    # Prompt (VERY IMPORTANT)
    prompt = f"""
You are a helpful AI assistant.

Answer the question based ONLY on the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{query}

Answer:
"""

    # Call LLM (Groq)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content