from pydantic import BaseModel
from typing import List, Optional

class IngestPDFResponse(BaseModel):
    document_id: str
    chunks: int

class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 5

class RetrievedChunk(BaseModel):
    document_id: str
    chunk_id: str
    text: str
    score: float
    page: Optional[int] = None

class RetrieveResponse(BaseModel):
    results: List[RetrievedChunk]