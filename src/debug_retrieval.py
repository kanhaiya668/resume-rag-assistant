import sys

from retriever import Retriever

DEFAULT_QUESTION = "How did Kanhaiya handle class imbalance in the fraud detection project?"

question = " ".join(sys.argv[1:]) or DEFAULT_QUESTION
print(f"Q: {question}")

for i, chunk in enumerate(Retriever(k=4).retrieve(question), start=1):
    print(f"\n--- Chunk {i} | score {chunk['score']} | {chunk['source']} ---")
    print(chunk["content"])