import os

from rag_chain import RAGAssistant

TEST_QUESTIONS = [
    "What is Kanhaiya's educational background?",
    "What are Kanhaiya's technical skills?",
    "What R² did the house price prediction model achieve?",
    "Explain the customer segmentation and recommendation project.",
    "What accuracy metrics did the demand forecasting model achieve?",
    "Which model was used for loan approval and how is it explained?",
    "Does Kanhaiya know deep learning?",
    "What salary does Kanhaiya expect?",
    "Kanhaiya ke projects kaun kaun se hain?",
    "How can I contact Kanhaiya?",
]

if __name__ == "__main__":
    assistant = RAGAssistant()
    for i, question in enumerate(TEST_QUESTIONS, start=1):
        result = assistant.ask(question)
        sources = sorted({os.path.basename(s["source"]) for s in result["sources"]})
        print(f"\n[{i}] Q: {question}\nA: {result['answer']}\nSources: {sources}")