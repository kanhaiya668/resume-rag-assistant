from langchain_text_splitters import RecursiveCharacterTextSplitter

from document_loader import load_documents


def chunk_documents(documents, chunk_size=800, chunk_overlap=150):  
    """
    Documents ko chhote overlapping chunks mein todta hai.
    Har chunk apna metadata (source file ka naam) saath rakhta hai.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)


if __name__ == "__main__":
    docs = load_documents()
    chunks = chunk_documents(docs)

    print(f"Documents: {len(docs)} -> Chunks: {len(chunks)}")

    avg_len = sum(len(c.page_content) for c in chunks) / len(chunks)
    print(f"Average chunk length: {avg_len:.0f} characters")

    print("\n--- Sample chunk ---")
    print("Source:", chunks[5].metadata["source"])
    print(chunks[5].page_content)