import os
import uuid
import zipfile
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename

upload_bp = Blueprint("upload", __name__)

ALLOWED_EXTENSIONS = {".c", ".h", ".cpp", ".hpp", ".rs", ".toml", ".json", ".lock", ".txt", ".md"}


def _build_file_tree(root_dir, original_root=None):
    """Recursively build a directory tree as nested dicts.

    All ``path`` values are relative to *original_root* so they can be used
    directly with the per-file API endpoint. When called externally, omit
    ``original_root`` — it defaults to ``root_dir``.
    """
    if original_root is None:
        original_root = root_dir
    tree = []
    try:
        entries = sorted(os.listdir(root_dir))
    except OSError:
        return tree

    for entry in entries:
        full_path = os.path.join(root_dir, entry)
        if entry.startswith(".") or entry == "__pycache__":
            continue
        # Path always relative to the top-level root
        rel_path = os.path.relpath(full_path, original_root)
        if os.path.isdir(full_path):
            tree.append({
                "name": entry,
                "type": "directory",
                "path": rel_path,
                "children": _build_file_tree(full_path, original_root),
            })
        elif os.path.isfile(full_path):
            ext = os.path.splitext(entry)[1].lower()
            tree.append({
                "name": entry,
                "type": "file",
                "path": rel_path,
                "size": os.path.getsize(full_path),
                "language": _guess_language(ext, entry),
            })
    return tree


def _compute_stats(tree):
    """Compute file/language statistics from a directory tree."""
    file_count = 0
    dir_count = 0
    extensions = {}

    def walk(nodes):
        nonlocal file_count, dir_count
        for node in nodes:
            if node["type"] == "directory":
                dir_count += 1
                walk(node.get("children", []))
            else:
                file_count += 1
                lang = node.get("language", "other")
                extensions[lang] = extensions.get(lang, 0) + 1

    walk(tree)
    return {
        "file_count": file_count,
        "dir_count": dir_count,
        "languages": extensions,
    }


def _guess_language(ext, filename):
    lang_map = {
        ".c": "c", ".h": "c",
        ".cpp": "cpp", ".hpp": "cpp", ".cc": "cpp", ".cxx": "cpp",
        ".rs": "rust",
        ".toml": "toml",
        ".json": "json",
        ".lock": "lock",
        ".txt": "text",
        ".md": "markdown",
    }
    return lang_map.get(ext, "other")


@upload_bp.route("/zip", methods=["POST"])
def upload_zip():
    """Upload a ZIP file containing a C/C++ code repository."""
    if "file" not in request.files:
        return jsonify({"code": 400, "message": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"code": 400, "message": "Empty filename"}), 400

    if not file.filename.lower().endswith(".zip"):
        return jsonify({"code": 400, "message": "Only .zip files are accepted"}), 400

    project_id = uuid.uuid4().hex[:12]
    project_dir = os.path.join(current_app.config["UPLOAD_FOLDER"], project_id)
    source_dir = os.path.join(project_dir, "source")
    os.makedirs(source_dir, exist_ok=True)

    # Save uploaded zip
    zip_path = os.path.join(project_dir, "uploaded.zip")
    file.save(zip_path)

    # Extract zip
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Strip common top-level wrapper directory
            members = zf.namelist()
            prefix = _find_common_prefix(members)

            # Derive project name: prefer common zip prefix, fall back to
            # the zip filename stem (e.g. "shared__541f4e547bdb" from
            # "shared__541f4e547bdb.zip").
            project_name = (prefix.rstrip("/") if prefix else
                           os.path.splitext(file.filename)[0] if file.filename else
                           project_id)
            # Save project name so downstream engine stages can use it
            _save_original_path(project_dir, project_name)

            for member in members:
                if member.endswith("/"):
                    continue
                rel_path = member
                if prefix and member.startswith(prefix):
                    rel_path = member[len(prefix):]
                target = os.path.join(source_dir, secure_filename_path(rel_path))
                if any(part.startswith(".") for part in rel_path.split("/")):
                    continue
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with zf.open(member) as src:
                    with open(target, "wb") as dst:
                        dst.write(src.read())
    except zipfile.BadZipFile:
        return jsonify({"code": 400, "message": "Invalid ZIP file"}), 400

    tree = _build_file_tree(source_dir)
    stats = _compute_stats(tree)

    return jsonify({
        "code": 200,
        "message": "OK",
        "data": {
            "project_id": project_id,
            "project_name": project_name,
            "file_tree": tree,
            "stats": stats,
        }
    })


def _find_common_prefix(members):
    """Find common root folder in zip members (if all files are under one top-level dir)."""
    entries_with_slash = set()    # e.g. "foo/"
    entries_without_slash = set()  # e.g. "foo" (bare dir entry) or "README.md"
    for m in members:
        stripped = m.rstrip("/")
        if not stripped:
            continue
        if "/" in stripped:
            entries_with_slash.add(stripped.split("/")[0] + "/")
        else:
            entries_without_slash.add(stripped)
    # If "foo/" exists, then the bare "foo" (from a directory entry) is redundant
    resolved = set(entries_with_slash)
    for entry in entries_without_slash:
        if entry + "/" not in entries_with_slash:
            resolved.add(entry)
    if len(resolved) == 1:
        prefix = resolved.pop()
        if prefix.endswith("/"):
            return prefix
    return None


def secure_filename_path(rel_path):
    """Build a safe path from relative zip entry, preventing directory traversal."""
    parts = rel_path.replace("\\", "/").split("/")
    safe_parts = []
    for p in parts:
        if p in ("", ".", ".."):
            continue
        safe_parts.append(secure_filename(p))
    return os.path.join(*safe_parts) if safe_parts else "unknown"


ORIGINAL_PATH_FILE = "original_path.txt"


def _save_original_path(project_dir: str, project_name: str) -> None:
    """Persist the extracted project name so downstream stages can use it."""
    with open(os.path.join(project_dir, ORIGINAL_PATH_FILE), "w") as f:
        f.write(project_name)


def get_project_name(project_dir: str) -> str:
    """Read the project name saved during upload, or derive from the directory name."""
    path_file = os.path.join(project_dir, ORIGINAL_PATH_FILE)
    if os.path.isfile(path_file):
        with open(path_file) as f:
            return f.read().strip()
    return os.path.basename(project_dir.rstrip("/"))
