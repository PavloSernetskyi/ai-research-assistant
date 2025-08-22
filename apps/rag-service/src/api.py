from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from .schemas import IngestPDFResponse, RetrieveRequest, RetrieveResponse, RetrievedChunk
from .ingestion import ingest_pdf_local
from .retrieval import add_records, retrieve
import os, uuid

app = FastAPI(title="RAG Service")

# CORS for local dev; tighten later in prod
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/ingest/pdf", response_model=IngestPDFResponse)
async def ingest_pdf(file: UploadFile = File(...)):
    os.makedirs("uploads", exist_ok=True)
    path = f"uploads/{uuid.uuid4()}_{file.filename}"
    with open(path, "wb") as f:
        f.write(await file.read())

    doc_id, records, vecs = ingest_pdf_local(path)
    add_records(records, vecs)
    return IngestPDFResponse(document_id=doc_id, chunks=len(records))

@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve_api(req: RetrieveRequest):
    hits = retrieve(req.query, req.top_k)
    results = [
        RetrievedChunk(
            document_id=h["document_id"],
            chunk_id=h["chunk_id"],
            text=h["text"][:1200],
            score=h["score"],
            page=h.get("page"),
        )
        for h in hits
    ]
    return RetrieveResponse(results=results)