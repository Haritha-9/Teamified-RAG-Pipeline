import fitz

def load_pdf(pdf_path: str):
    doc = fitz.open(pdf_path)
    pages = []

    for page_number, page in enumerate(doc):
        text = page.get_text()
        pages.append({
            "page": page_number + 1,
            "content": text
        })

    return pages