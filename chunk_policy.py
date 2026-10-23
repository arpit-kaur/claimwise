"""Load the insurance policy PDF and split it into chunks for RAG."""

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

POLICY_FILE = Path(__file__).parent / "data" / "policy.pdf"


def load_policy():
    """Read the PDF and return one Document per page."""
    reader = PdfReader(POLICY_FILE)
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append(Document(page_content=text, metadata={"page": page_number}))

    return pages


def split_into_chunks(pages, chunk_size=800, chunk_overlap=100):
    """Split pages into smaller, overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(pages)


if __name__ == "__main__":
    pages = load_policy()
    print("Pages loaded:", len(pages))
    print("Characters on page 1:", len(pages[0].page_content))

    chunks = split_into_chunks(pages)
    print("Chunks created:", len(chunks))

    print("\n--- Chunk 1 ---")
    print(chunks[0].page_content)
    print("Metadata:", chunks[0].metadata)

    print("\n--- Chunk 2 ---")
    print(chunks[1].page_content)
    print("Metadata:", chunks[1].metadata)

    small = split_into_chunks(pages, chunk_size=300, chunk_overlap=50)
    large = split_into_chunks(pages, chunk_size=2000, chunk_overlap=200)
    print("\nWith chunk_size=300:", len(small), "chunks")
    print("With chunk_size=2000:", len(large), "chunks")

    keyword = "emergency"
    matches = [chunk for chunk in chunks if keyword in chunk.page_content.lower()]
    print(f"\nChunks containing the word '{keyword}':", len(matches))