ALLOWED_TYPES = [".pdf", ".txt", ".docx"]

def validate_file(path):
    import os
    ext = os.path.splitext(path)[1].lower()
    if ext not in ALLOWED_TYPES:
        raise ValueError("Unsupported file type")
    return True
