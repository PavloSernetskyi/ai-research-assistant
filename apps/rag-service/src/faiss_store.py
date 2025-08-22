import faiss, json, os
from typing import List, Dict
import numpy as np

class FaissStore:
    def __init__(self, dim: int, index_path: str):
        self.dim = dim
        self.index_path = index_path
        self.meta_path = self.index_path + ".meta.json"

        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            self.index = faiss.IndexFlatIP(dim)

        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.meta: List[Dict] = json.load(f)
        else:
            self.meta = []

    def add(self, records: List[Dict], vectors: np.ndarray) -> None:
        """
        records: [{chunk_id, document_id, text, page}]
        vectors: (N, dim) float32
        """
        assert len(records) == vectors.shape[0], "records and vectors mismatch"
        self.index.add(vectors)
        self.meta.extend(records)
        self._persist()

    def search(self, qvec: np.ndarray, top_k: int = 5) -> List[Dict]:
        D, I = self.index.search(qvec, top_k)
        results: List[Dict] = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            m = self.meta[idx]
            results.append({**m, "score": float(score)})
        return results

    def _persist(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)