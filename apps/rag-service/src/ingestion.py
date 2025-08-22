import uuid
import fitz  # PyMuPDF
from .chunking import chunk_text
from .embeddings import embed_texts

def extract_text_from_pdf(path: str) -> str:
    doc = fitz.open(path)
    pages = [p.get_text() for p in doc]
    return "\n".join(pages)

def ingest_pdf_local(path: str):
    text = extract_text_from_pdf(path)
    chunks = chunk_text(text)
    vecs = embed_texts(chunks)
    # assemble records with UUIDs
    doc_id = str(uuid.uuid4())
    records = [
        {"chunk_id": str(uuid.uuid4()), "document_id": doc_id, "text": c, "page": None}
        for c in chunks
    ]
    return doc_id, records, vecs