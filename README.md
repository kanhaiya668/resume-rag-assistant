# 💼 Resume RAG Assistant


🔗 **[Live Demo](https://resume-rag-assistant-izalayzrgdmniwy5j7hmch.streamlit.app/)
** · [GitHub](https://github.com/kanhaiya668/resume-rag-assistant)


An AI assistant that answers recruiter questions about Kanhaiya Bhardwaj —
his projects, skills, and background — grounded entirely in his resume and
project documentation, using Retrieval-Augmented Generation (RAG).

## Why RAG, not just an LLM

A plain LLM would either know nothing about a specific person, or risk
making up plausible-sounding but false details. This assistant retrieves
relevant chunks from real source documents before generating an answer,
and refuses to answer when nothing relevant is found — reducing
hallucination and keeping every answer traceable to a source.

## How it works

1. **Ingest** — resume (PDF) and project READMEs (text) are loaded and
   split into overlapping chunks.
2. **Embed** — each chunk is converted to a 384-dim vector using a local
   sentence-transformers model (`all-MiniLM-L6-v2`) — no API cost.
3. **Index** — vectors are stored in a FAISS index for fast similarity search.
4. **Retrieve** — a query is embedded and matched against the index;
   only chunks above a relevance threshold are kept.
5. **Generate** — retrieved chunks + the question are sent to an LLM
   (Groq, `openai/gpt-oss-120b`) with a prompt that restricts it to
   answering only from the provided context.

## Tech stack

Python · LangChain · FAISS · sentence-transformers · Groq API · Streamlit

## Run locally

\`\`\`bash
git clone https://github.com/kanhaiya668/resume-rag-assistant.git
cd resume-rag-assistant
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
\`\`\`

Create a `.env` file with:
\`\`\`
GROQ_API_KEY=your_key_here
\`\`\`

Build the index, then launch the app:
\`\`\`bash
python src/vector_store.py
streamlit run app.py
\`\`\`

## Project structure

\`\`\`
resume-rag-assistant/
├── app.py                  # Streamlit chat UI
├── data/                   # Source documents (resume, project docs)
├── src/
│   ├── document_loader.py  # Loads PDF/text files
│   ├── text_splitter.py    # Chunks documents
│   ├── vector_store.py     # Builds/loads the FAISS index
│   ├── retriever.py        # Semantic search over the index
│   └── rag_chain.py        # Retrieval + Groq generation
└── requirements.txt
\`\`\`

## Live demo: 🔗 **[Try it here](https://resume-rag-assistant-izalayzrgdmniwy5j7hmch.streamlit.app/)**