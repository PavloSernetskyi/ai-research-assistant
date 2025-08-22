from sentence_transformers import SentenceTransformer
import numpy as np

_model = None

def get_model():
    global _model
    if _model is None:
        # solid default; you can switch later to bge-m3, gte-large, etc.
        _model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
    return _model

def embed_texts(texts: list[str]) -> np.ndarray:
    model = get_model()
    vecs = model.encode(texts, normalize_embeddings=True)
    return np.asarray(vecs, dtype="float32")