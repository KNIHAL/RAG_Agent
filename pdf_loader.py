import fitz  # PyMuPDF

def load_pdf_chunks(pdf_path: str, chunk_size: int = 700, overlap: int = 100):
    """
    Load a PDF and split its full text into overlapping chunks.
    """
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()

    chunks = []
    start = 0
    while start < len(full_text):
        end = start + chunk_size
        chunk = full_text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks
