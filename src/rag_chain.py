from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from retriever import Retriever

import unicodedata

_CHAR_MAP = str.maketrans({
    "\u2011": "-",
    "\u2013": "-",
    "\u2014": "-",
    "\u2018": "'", "\u2019": "'",
    "\u201c": '"', "\u201d": '"',
    "\u00a0": " ", "\u2009": " ",
})


def clean_text(text):
    """LLM ke output se typographic Unicode characters hataakar plain ASCII banata hai."""
    text = unicodedata.normalize("NFKC", text)
    return text.translate(_CHAR_MAP)

load_dotenv()

LLM_MODEL = "openai/gpt-oss-120b"
MIN_SCORE = 0.25

NO_INFO_MESSAGE = (
    "I couldn't find that in Kanhaiya's resume or project documents. "
    "Try asking about his projects, skills, education, or career goals."
)

SYSTEM_PROMPT = """You are an assistant that answers questions about Kanhaiya Bhardwaj, \
a fresher Data Scientist, for recruiters and hiring managers.

Rules:
- Answer ONLY from the context below. Never use outside knowledge or guess.
- Do not add details, qualifiers or assumptions (for example dates being "expected") that are not stated in the context.
- If the context does not contain the answer, say you don't have that information.
- Always refer to him in the third person ("Kanhaiya", "he").
- Be concise and professional. Include concrete numbers (metrics, dataset sizes) when the context has them.
- Reply in the same language as the question.
- Never state or estimate a salary, compensation figure, or any specific number unless that exact number appears in the context. If asked about compensation and it is not in the context, say this is not something you have information on.


Context:
{context}"""


class RAGAssistant:
    """Retrieval + Groq LLM ko jodkar sawaal ka grounded answer deta hai."""

    def __init__(self):
        self.retriever = Retriever(k=6)
        llm = ChatGroq(model=LLM_MODEL, temperature=0)
        prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT), ("human", "{question}")]
        )
        self.chain = prompt | llm

    def ask(self, question):
        chunks = [
            c for c in self.retriever.retrieve(question) if c["score"] >= MIN_SCORE
        ]
        if not chunks:
            return {"answer": NO_INFO_MESSAGE, "sources": []}

        context = "\n\n".join(f"[{c['source']}]\n{c['content']}" for c in chunks)
        response = self.chain.invoke({"context": context, "question": question})
        return {"answer": clean_text(response.content), "sources": chunks}

if __name__ == "__main__":
    assistant = RAGAssistant()
    questions = [
        "How did Kanhaiya handle class imbalance in the fraud detection project?",
        "What is Kanhaiya's educational background?",
        "What is the capital of France?",
    ]
    for q in questions:
        result = assistant.ask(q)
        print(f"\nQ: {q}\nA: {result['answer']}")
        print("Sources:", sorted({s["source"] for s in result["sources"]}))