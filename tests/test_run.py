from src.loader import load_pdf
from src.chunker import chunk_documents
from src.vector_store import build_or_load_vector_store
from src.intent_classifier import classify_intent
from src.answer_chain import build_answer_chain


def simple_run_test():

    print("Running Simple RAG Test")
    print("------------------------")

    # Use a small PDF for testing
    pdf_path = "sample_test.pdf"

    # Create a small test PDF dynamically
    import fitz
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Jose Rizal was a Filipino nationalist.")
    doc.save(pdf_path)
    doc.close()

    # 1️⃣ Load
    pages = load_pdf(pdf_path)
    print("Loaded pages:", len(pages))

    # 2️⃣ Chunk
    documents = chunk_documents(pages)
    print("Chunks created:", len(documents))

    # 3️⃣ Vector Store
    vector_store = build_or_load_vector_store(documents)
    print("Vector store built")

    # 4️⃣ Retrieval
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    query = "Who was Jose Rizal?"
    retrieved_docs = retriever.invoke(query)

    print("Retrieved documents:", len(retrieved_docs))

    # 5️⃣ Intent
    intent = classify_intent(query)
    print("Detected intent:", intent)

    # 6️⃣ Answer
    answer_chain = build_answer_chain()

    context = "\n".join(
        [doc.page_content for doc in retrieved_docs]
    )

    response = answer_chain.invoke({
        "context": context,
        "question": query,
        "intent": intent
    })

    print("\nFinal Answer:\n", response)

    print("\n✅ Simple run test completed successfully")


if __name__ == "__main__":
    simple_run_test()