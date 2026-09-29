from vector_store import load_vector_store


class Retriever:
    """FAISS index se sawaal ke sabse relevant chunks dhundhta hai."""

    def __init__(self, k=4):
        self.k = k
        self.store = load_vector_store()

    def retrieve(self, query):
        """Top-k chunks return karta hai: content, source file aur relevance score."""
        results = self.store.similarity_search_with_score(query, k=self.k)
        return [
            {
                "content": doc.page_content,
                "source": doc.metadata["source"],
                # vectors normalized hain: squared L2 distance -> cosine similarity
                "score": round(1 - float(distance) / 2, 3),
            }
            for doc, distance in results
        ]


if __name__ == "__main__":
    retriever = Retriever()
    queries = [
        "How was class imbalance handled in the fraud project?",
        "What is Kanhaiya's educational background?",
        "What is the capital of France?",
    ]
    for q in queries:
        print(f"\nQ: {q}")
        for r in retriever.retrieve(q):
            preview = r["content"][:80].replace("\n", " ")
            print(f"  {r['score']:.3f} | {r['source']} | {preview}")