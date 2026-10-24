"""Turn policy chunks into embeddings, store them in Chroma, and search by meaning."""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from chunk_policy import load_policy, split_into_chunks

STORE_DIR = Path(__file__).parent / "vector_store"
embeddings = OllamaEmbeddings(model="nomic-embed-text")


def build_store():
    """Embed every chunk and save them in a Chroma vector database."""
    chunks = split_into_chunks(load_policy())
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(STORE_DIR),
        collection_name="policy",
    )
    return len(chunks)


def load_store():
    """Open the saved vector database."""
    return Chroma(
        persist_directory=str(STORE_DIR),
        embedding_function=embeddings,
        collection_name="policy",
    )


def search_policy(question, k=3):
    """Return the k chunks closest in meaning to the question, with scores."""
    store = load_store()
    return store.similarity_search_with_score(question, k=k)


if __name__ == "__main__":
    if STORE_DIR.exists():
        print("Vector database already exists. Using it.")
    else:
        print("Building the vector database. This can take a minute...")
        count = build_store()
        print("Stored", count, "chunks.")

    questions = [
        "How much does it cost to go to the ER?",
        "What is the deductible?",
        "Is there a limit on how much I pay in a year?",
    ]

    for question in questions:
        print("\n" + "=" * 70)
        print("QUESTION:", question)
        results = search_policy(question)
        for rank, (chunk, score) in enumerate(results, start=1):
            preview = chunk.page_content[:150].replace("\n", " ")
            print(f"\n  {rank}. Page {chunk.metadata['page']} | distance {score:.3f}")
            print("    ", preview, "...")