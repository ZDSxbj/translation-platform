"""ZIP file utilities."""

import os
import zipfile
import io


def zip_directory(source_dir: str) -> io.BytesIO:
    """Create an in-memory ZIP from a directory tree."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(source_dir):
            for fn in files:
                full_path = os.path.join(root, fn)
                arcname = os.path.relpath(full_path, source_dir)
                zf.write(full_path, arcname)
    buf.seek(0)
    return buf


def extract_zip(zip_data: bytes, target_dir: str):
    """Extract ZIP bytes to a target directory."""
    os.makedirs(target_dir, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
        zf.extractall(target_dir)
