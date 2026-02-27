import os
from langchain_core.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from src.config import OPENAI_API_KEY

INDEX_PATH = "faiss_index"

def build_or_load_vector_store(documents):
    embeddings = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY,
        model="text-embedding-3-small"
    )

    if os.path.exists(INDEX_PATH):
        return FAISS.load_local(
            INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local(INDEX_PATH)
    return vector_store