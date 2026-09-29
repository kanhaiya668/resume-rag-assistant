import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader

def load_documents(data_folder="data"):
    """
    data/ folder ke andar sabhi PDF aur text files padhta hai
    aur unka content ek list mein return karta hai.
    """
    documents = []

    for filename in os.listdir(data_folder):
        file_path = os.path.join(data_folder, filename)

        if filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())

        elif filename.endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
            documents.extend(loader.load())

    return documents


if __name__ == "__main__":
    docs = load_documents()
    print(f"Total documents loaded: {len(docs)}\n")
    for d in docs:
        print(d.metadata["source"], "->", len(d.page_content), "characters")