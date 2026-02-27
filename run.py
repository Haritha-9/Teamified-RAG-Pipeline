import os

from src.loader import load_pdf
from src.chunker import chunk_documents
from src.vector_store import build_or_load_vector_store
from src.intent_classifier import classify_intent
from src.answer_chain import build_answer_chain
from src.config import TOP_K


def main():

    print("AI Context Builder (RAG System)")
    print("----------------------------------")

    # Ask user for PDF path
    pdf_path = input("C:/Users/harharitha/Downloads/PHILIPPINE-HISTORY-SOURCE-BOOK-FINAL-SEP022021.pdf").strip()

    if not os.path.exists(pdf_path):
        print("Error: File not found.")
        return

    print("\nLoading and indexing PDF...")

    pages = load_pdf(pdf_path)
    documents = chunk_documents(pages)
    vector_store = build_or_load_vector_store(documents)

    retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})
    answer_chain = build_answer_chain()

    print("System ready.\n")

    while True:
        query = input("Enter your question (type 'exit' to quit): ")

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
        print("\nAnswer:\n", response)
        print("\nSource Pages:",
              [doc.metadata["page"] for doc in retrieved_docs])
        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()