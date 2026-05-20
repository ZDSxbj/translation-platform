"""File and repository operations utility."""

import os
import shutil


class FileService:
    """Utility for project file operations."""

    @staticmethod
    def get_source_dir(upload_folder: str, project_id: str) -> str:
        return os.path.join(upload_folder, project_id, "source")

    @staticmethod
    def get_output_dir(output_folder: str, session_id: str) -> str:
        return os.path.join(output_folder, session_id)

    @staticmethod
    def ensure_dir(path: str):
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def copy_tree(src: str, dst: str):
        """Copy directory tree from src to dst."""
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst, symlinks=True)

    @staticmethod
    def remove_dir(path: str):
        """Remove a directory tree."""
        if os.path.exists(path):
            shutil.rmtree(path)
