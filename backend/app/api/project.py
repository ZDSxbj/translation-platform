import os
import json
from flask import Blueprint, request, jsonify, current_app
from app.services.path_service import PathService

project_bp = Blueprint("project", __name__)


@project_bp.route("/<project_id>/tree", methods=["GET"])
def get_tree(project_id):
    """Get the directory tree of a project."""
    project_dir = _get_project_source_dir(project_id)
    if not os.path.isdir(project_dir):
        return jsonify({"code": 404, "message": "Project not found"}), 404

    from app.api.upload import _build_file_tree  # Reuse the tree builder
    tree = _build_file_tree(project_dir)
    return jsonify({"code": 200, "data": {"file_tree": tree}})


@project_bp.route("/<project_id>/file", methods=["GET"])
def get_file(project_id):
    """Get the content of a specific file in the project."""
    file_path = request.args.get("path", "")
    if not file_path:
        return jsonify({"code": 400, "message": "Missing 'path' parameter"}), 400

    project_dir = _get_project_source_dir(project_id)
    full_path = os.path.normpath(os.path.join(project_dir, file_path))

    # Security: ensure path is within project directory
    if not full_path.startswith(os.path.normpath(project_dir)):
        return jsonify({"code": 403, "message": "Path traversal denied"}), 403

    if not os.path.isfile(full_path):
        return jsonify({"code": 404, "message": "File not found"}), 404

    # Detect file type for syntax highlighting
    ext = os.path.splitext(full_path)[1].lower()

    try:
        with open(full_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception:
        # Try as binary
        try:
            with open(full_path, "rb") as f:
                content = f.read().decode("utf-8", errors="replace")
        except Exception:
            return jsonify({"code": 400, "message": "Cannot read file"}), 400

    return jsonify({
        "code": 200,
        "data": {
            "path": file_path,
            "content": content,
            "language": _ext_to_language(ext),
            "size": os.path.getsize(full_path),
        }
    })


@project_bp.route("/<project_id>/stats", methods=["GET"])
def get_stats(project_id):
    """Get project statistics."""
    project_dir = _get_project_source_dir(project_id)
    if not os.path.isdir(project_dir):
        return jsonify({"code": 404, "message": "Project not found"}), 404

    from app.api.upload import _build_file_tree, _compute_stats
    tree = _build_file_tree(project_dir)
    stats = _compute_stats(tree)

    return jsonify({"code": 200, "data": stats})


@project_bp.route("/<project_id>/compile_commands", methods=["GET"])
def get_compile_commands(project_id):
    """Check if compile_commands.json exists and return its path info."""
    project_dir = _get_project_source_dir(project_id)
    cc_path = os.path.join(project_dir, "compile_commands.json")
    exists = os.path.isfile(cc_path)

    return jsonify({
        "code": 200,
        "data": {
            "has_compile_commands": exists,
            "path": "compile_commands.json" if exists else None,
        }
    })


@project_bp.route("/<project_id>/analyze", methods=["GET"])
def analyze_project(project_id):
    """Analyze project type and compile_commands.json health.

    Returns project classification (ohos / standard_c / unknown),
    dependency information, and recommendations for configuration.
    """
    project_dir = _get_project_source_dir(project_id)
    if not os.path.isdir(project_dir):
        return jsonify({"code": 404, "message": "Project not found"}), 404

    ps = PathService(project_dir)
    result = ps.analyze()

    return jsonify({"code": 200, "data": result})


@project_bp.route("/<project_id>/fix-paths", methods=["POST"])
def fix_project_paths(project_id):
    """Relativize absolute paths in compile_commands.json."""
    project_dir = _get_project_source_dir(project_id)
    if not os.path.isdir(project_dir):
        return jsonify({"code": 404, "message": "Project not found"}), 404

    ps = PathService(project_dir)
    result = ps.relativize_paths()

    if result["success"]:
        return jsonify({"code": 200, "data": result})
    else:
        return jsonify({"code": 400, "message": result["message"]}), 400


def _get_project_source_dir(project_id):
    return os.path.join(current_app.config["UPLOAD_FOLDER"], project_id, "source")


def _ext_to_language(ext):
    return {
        ".c": "c", ".h": "c",
        ".cpp": "cpp", ".hpp": "cpp", ".cc": "cpp", ".cxx": "cpp",
        ".rs": "rust",
        ".toml": "toml",
        ".json": "json",
        ".txt": "text",
        ".md": "markdown",
    }.get(ext, "text")
