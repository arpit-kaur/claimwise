"""See what embeddings look like and compare meanings."""

import numpy as np
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")


def cosine_similarity(a, b):
    """Return how similar two embeddings are (closer to 1 = more similar)."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


question = embeddings.embed_query("How much does it cost to go to the ER?")
emergency = embeddings.embed_query("Emergency room care: copay per visit")
office = embeddings.embed_query("Primary care office visit to treat an injury or illness")
weather = embeddings.embed_query("It will snow in Winnipeg this weekend")

print("Numbers in one embedding:", len(question))
print("First 5 numbers:", [round(x, 3) for x in question[:5]])

print("\nSimilarity to the ER question:")
print("Emergency room text:", round(cosine_similarity(question, emergency), 3))
print("Office visit text:  ", round(cosine_similarity(question, office), 3))
print("Weather text:       ", round(cosine_similarity(question, weather), 3))