from pathlib import Path


def load_pdf_texts(pdf_dir: str) -> dict:
    """
    Loads extracted PDF texts.

    Input:
        pdf_dir — directory containing pre-extracted text files

    Output:
        dict[doc_id] -> full text
    """
    docs = {}
    for file in Path(pdf_dir).glob("*.txt"):
        docs[file.stem] = file.read_text(encoding="utf-8")
    return docs
