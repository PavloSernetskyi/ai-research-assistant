import numpy as np
from .embeddings import embed_texts
from .faiss_store import FaissStore
from .settings import settings

_store: FaissStore | None = None

def get_store(dim: int = 1024) -> FaissStore:
    global _store
    if _store is None:
        _store = FaissStore(dim=dim, index_path=settings.index_path)
    return _store

def add_records(records, vecs):
    get_store().add(records, vecs)

def retrieve(query: str, top_k: int = 5):
    qvec = embed_texts([query])  # (1, dim)
    return get_store().search(qvec, top_k)