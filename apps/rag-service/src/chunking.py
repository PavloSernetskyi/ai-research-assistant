def chunk_text(text: str, max_tokens: int = 450, overlap: int = 60) -> list[str]:
    """
    Simple word-count-based chunker for MVP.
    Replace with token-aware chunker later (e.g., tiktoken).
    """
    words = text.split()
    chunks, step = [], max_tokens - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i+max_tokens]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks