from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.config import CHUNK_SIZE, CHUNK_OVERLAP
def chunk_documents(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP)
    documents = []
    for page in pages:
        chunks = splitter.split_text(page["content"])
        for chunk in chunks:
            documents.append(
                Document(page_content=chunk, metadata={"page": page["page"]}))
    return documents