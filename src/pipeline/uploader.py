import os

def upload_file(file_path, storage_dir="data/raw"):
    os.makedirs(storage_dir, exist_ok=True)
    dest = os.path.join(storage_dir, os.path.basename(file_path))
    with open(file_path, "rb") as src, open(dest, "wb") as dst:
        dst.write(src.read())
    return dest
