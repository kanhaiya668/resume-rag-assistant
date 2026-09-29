import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).parent / "src"))
from rag_chain import RAGAssistant

st.set_page_config(page_title="Kanhaiya Bhardwaj — Resume Assistant", page_icon="💼")

st.title("💼 Ask about Kanhaiya Bhardwaj")
st.caption(
    "AI assistant built on a RAG pipeline (FAISS + Groq) over Kanhaiya's "
    "resume and project documentation. Ask about his skills, projects, or background."
)


if st.session_state.get("messages"):
    if st.button("🔄 Clear chat"):
        st.session_state.messages = []
        st.rerun()


@st.cache_resource
def get_assistant():
    return RAGAssistant()


assistant = get_assistant()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for s in message["sources"]:
                    st.caption(f"{s['source']} · relevance {s['score']}")

EXAMPLE_QUESTIONS = [
    "What projects has Kanhaiya built?",
    "What are his technical skills?",
    "How did he handle class imbalance in the fraud project?",
]

if not st.session_state.messages:
    st.write("**Try asking:**")
    cols = st.columns(len(EXAMPLE_QUESTIONS))
    for col, q in zip(cols, EXAMPLE_QUESTIONS):
        if col.button(q):
            st.session_state.pending_question = q
            st.rerun()

question = st.chat_input("Ask a question...")
if not question and "pending_question" in st.session_state:
    question = st.session_state.pop("pending_question")


if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = assistant.ask(question)
                except Exception:
                    result = {
                        "answer": "Something went wrong while generating a response. Please try again in a moment.",
                        "sources": [],
                }
            st.markdown(result["answer"])
        if result["sources"]:
            with st.expander("Sources"):
                for s in result["sources"]:
                    st.caption(f"{s['source']} · relevance {s['score']}")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        }
    )