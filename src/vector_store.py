from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from document_loader import load_documents
from text_splitter import chunk_documents

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_DIR = Path("vectorstore")


def get_embeddings():
    """Text ko vectors mein badalne wala model load karta hai."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vector_store(index_dir=INDEX_DIR):
    """Documents load -> chunk -> embed -> FAISS index banakar disk pe save karta hai."""
    documents = load_documents()
    chunks = chunk_documents(documents)
    store = FAISS.from_documents(chunks, get_embeddings())
    store.save_local(str(index_dir))
    return store


def load_vector_store(index_dir=INDEX_DIR):
    """Pehle se bana hua index disk se load karta hai (dobara embed nahi karta)."""
    return FAISS.load_local(
        str(index_dir),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )



def get_vector_store(index_dir=INDEX_DIR):
    """Index disk pe mile to load karta hai, warna khud bana leta hai (fresh deploy ke liye)."""
    if index_dir.exists():
        return load_vector_store(index_dir)
    return build_vector_store(index_dir)


if __name__ == "__main__":
    store = build_vector_store()
    print(f"Index built: {store.index.ntotal} vectors, {store.index.d} dimensions")
    print(f"Saved to: {INDEX_DIR}/")

    print("\n--- Sanity check: semantic search ---")
    query = "How was class imbalance handled?"
    for result in store.similarity_search(query, k=2):
        print(f"\nSource: {result.metadata['source']}")
        print(result.page_content[:250])