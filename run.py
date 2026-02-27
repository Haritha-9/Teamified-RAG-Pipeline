import os

from src.loader import load_pdf
from src.chunker import chunk_documents
from src.vector_store import build_or_load_vector_store
from src.intent_classifier import classify_intent
from src.answer_chain import build_answer_chain
from src.config import TOP_K

import time


def main():

    print("AI Context Builder (RAG System)")
    print("----------------------------------")

    # Ask user for PDF path
    pdf_path = input("Enter full path to your PDF file: ").strip()
    start = time.time()
    pages = load_pdf(pdf_path)
    print("PDF loaded in", time.time() - start, "seconds")


    if not os.path.exists(pdf_path):
        print("Error: File not found.")
        return

    print("\nLoading and indexing PDF...")

    pages = load_pdf(pdf_path)
    documents = chunk_documents(pages)
    vector_store = build_or_load_vector_store(documents)

    start = time.time()
    documents = chunk_documents(pages)
    print("Chunking done in", time.time() - start, "seconds")

    

    retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})
    answer_chain = build_answer_chain()

    print("System ready.\n")

    while True:
        query = input("User Query (type 'exit' to quit): ")

        if query.lower() == "exit":
            break

        intent = classify_intent(query)
        retrieved_docs = retriever.invoke(query)

        context = "\n\n".join([
            f"(Page {doc.metadata['page']}) {doc.page_content}"
            for doc in retrieved_docs
        ])

        response = answer_chain.invoke({
            "context": context,
            "question": query,
            "intent": intent
        })

        print("\nDetected Intent:", intent)
        print("\nRetrieved Chunks:")
        for i, doc in enumerate(retrieved_docs):
            print(f"\n --{doc.page_content}")
        print("\nLLM Response:\n", response)

        print("\n" + "-" * 60 + "\n")
    main()
