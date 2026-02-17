def chunk_document(text: str, chunk_size: int = 500, overlap: int = 50):
    """
    Sliding-window chunking.

    Returns:
        List[str]
    """
    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks
