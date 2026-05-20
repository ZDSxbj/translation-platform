import uuid
from flask import Blueprint, request, jsonify, current_app
from app.services.pipeline_manager import get_pipeline_manager

translate_bp = Blueprint("translate", __name__)


@translate_bp.route("/start", methods=["POST"])
def start_translation():
    """Start a new translation session."""
    data = request.get_json(silent=True) or {}

    project_id = data.get("project_id")
    if not project_id:
        return jsonify({"code": 400, "message": "Missing 'project_id'"}), 400

    config = {
        "engine": data.get("engine", "his2trans"),
        "model": data.get("model", current_app.config.get("API_MODEL", "deepseek-v3.2")),
        "use_rag": data.get("use_rag", False),
        "max_repair": int(data.get("max_repair", 5)),
        "api_key": data.get("api_key", current_app.config.get("API_KEY", "")),
        "api_base_url": data.get("api_base_url", current_app.config.get("API_BASE_URL", "")),
        "api_max_tokens": int(data.get("api_max_tokens", current_app.config.get("API_MAX_TOKENS", 8192))),
        "api_temperature": float(data.get("api_temperature", current_app.config.get("API_TEMPERATURE", 0.0))),
        "his2trans_framework": current_app.config.get("HIS2TRANS_FRAMEWORK", ""),
        "his2trans_data": current_app.config.get("HIS2TRANS_DATA", ""),
        # Project-type-dependent fields
        "ohos_root": data.get("ohos_root", ""),
        "extra_includes": data.get("extra_includes", []),
    }

    session_id = str(uuid.uuid4())
    pipeline = get_pipeline_manager()
    pipeline.create_session(
        session_id=session_id,
        project_id=project_id,
        config=config,
        upload_folder=current_app.config["UPLOAD_FOLDER"],
        output_folder=current_app.config["OUTPUT_FOLDER"],
    )

    return jsonify({
        "code": 200,
        "message": "Translation session created",
        "data": {
            "session_id": session_id,
            "stages": pipeline.get_stages(),
        }
    })


@translate_bp.route("/<session_id>/state", methods=["GET"])
def get_state(session_id):
    """Get current pipeline state."""
    pipeline = get_pipeline_manager()
    state = pipeline.get_session_state(session_id)
    if state is None:
        return jsonify({"code": 404, "message": "Session not found"}), 404
    return jsonify({"code": 200, "data": state})


@translate_bp.route("/<session_id>/stage/<stage_id>/run", methods=["POST"])
def run_stage(session_id, stage_id):
    """Run a specific pipeline stage."""
    pipeline = get_pipeline_manager()
    try:
        result = pipeline.run_stage(session_id, stage_id)
        return jsonify({"code": 200, "data": result})
    except ValueError as e:
        return jsonify({"code": 400, "message": str(e)}), 400
    except Exception as e:
        return jsonify({"code": 500, "message": str(e)}), 500


@translate_bp.route("/<session_id>/stage/<stage_id>/result", methods=["GET"])
def get_stage_result(session_id, stage_id):
    """Get the result of a completed stage."""
    pipeline = get_pipeline_manager()
    result = pipeline.get_stage_result(session_id, stage_id)
    if result is None:
        return jsonify({"code": 404, "message": "Stage result not found"}), 404
    return jsonify({"code": 200, "data": result})


@translate_bp.route("/<session_id>/stage/<stage_id>/logs", methods=["GET"])
def get_stage_logs(session_id, stage_id):
    """Get logs for a stage."""
    pipeline = get_pipeline_manager()
    logs = pipeline.get_stage_logs(session_id, stage_id)
    if logs is None:
        return jsonify({"code": 404, "message": "Stage not found"}), 404
    return jsonify({"code": 200, "data": logs})


@translate_bp.route("/<session_id>/output/tree", methods=["GET"])
def get_output_tree(session_id):
    """Get the output directory tree after translation.

    Query params:
        subdir: Relative path within output (e.g. 'workspace/final_projects')
                Defaults to 'workspace/final_projects' to show only the final
                translated Rust project, not all intermediate artifacts.
    """
    pipeline = get_pipeline_manager()
    state = pipeline.get_session_state(session_id)
    if state is None:
        return jsonify({"code": 404, "message": "Session not found"}), 404

    output_path = state.get("output_path", "")
    if not output_path or not __import__("os").path.isdir(output_path):
        return jsonify({"code": 404, "message": "Output not yet available"}), 404

    subdir = request.args.get("subdir", "workspace/final_projects")
    target_path = _os.path.normpath(_os.path.join(output_path, subdir))
    # Security: ensure resolved path is still within output_path
    if not target_path.startswith(_os.path.normpath(output_path)):
        return jsonify({"code": 403, "message": "Path traversal denied"}), 403

    if not _os.path.isdir(target_path):
        return jsonify({"code": 200, "data": {"file_tree": [], "path": subdir}})

    from app.api.upload import _build_file_tree
    tree = _build_file_tree(target_path, original_root=_os.path.normpath(output_path))
    return jsonify({"code": 200, "data": {"file_tree": tree, "path": subdir}})


@translate_bp.route("/<session_id>/output/file", methods=["GET"])
def get_output_file(session_id):
    """Get a file from the translated output."""
    return _serve_workspace_file(session_id, "output")


@translate_bp.route("/<session_id>/report", methods=["GET"])
def get_report(session_id):
    """Get the translation report as JSON."""
    pipeline = get_pipeline_manager()
    report = pipeline.get_report(session_id)
    if report is None:
        return jsonify({"code": 404, "message": "Session not found"}), 404
    return jsonify({"code": 200, "data": report})


@translate_bp.route("/<session_id>/workspace/tree", methods=["GET"])
def get_workspace_tree(session_id):
    """Get the workspace directory tree, optionally scoped to a subdirectory.

    Query params:
        subdir: Relative path within workspace (e.g. 'skeletons/myproject')
    """
    pipeline = get_pipeline_manager()
    state = pipeline.get_session_state(session_id)
    if state is None:
        return jsonify({"code": 404, "message": "Session not found"}), 404

    output_path = state.get("output_path", "")
    workspace_path = _os.path.join(output_path, "workspace")

    subdir = request.args.get("subdir", "")
    if subdir:
        workspace_path = _os.path.normpath(_os.path.join(workspace_path, subdir))
        if not workspace_path.startswith(_os.path.normpath(_os.path.join(output_path, "workspace"))):
            return jsonify({"code": 403, "message": "Path traversal denied"}), 403

    if not _os.path.isdir(workspace_path):
        return jsonify({"code": 200, "data": {"file_tree": [], "path": subdir}})

    from app.api.upload import _build_file_tree
    workspace_root = _os.path.join(output_path, "workspace")
    tree = _build_file_tree(workspace_path, original_root=workspace_root)
    return jsonify({"code": 200, "data": {"file_tree": tree, "path": subdir}})


@translate_bp.route("/<session_id>/workspace/file", methods=["GET"])
def get_workspace_file(session_id):
    """Get a file from the workspace directory."""
    return _serve_workspace_file(session_id, "workspace")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

import os as _os


def _serve_workspace_file(session_id, root_type):
    """Serve a file from either 'output' or 'workspace' root.

    Args:
        session_id: Session identifier.
        root_type: Either 'output' (reads from output_path directly) or
                   'workspace' (reads from output_path/workspace/).
    """
    file_path = request.args.get("path", "")
    if not file_path:
        return jsonify({"code": 400, "message": "Missing 'path' parameter"}), 400

    pipeline = get_pipeline_manager()
    state = pipeline.get_session_state(session_id)
    if state is None:
        return jsonify({"code": 404, "message": "Session not found"}), 404

    output_path = state.get("output_path", "")
    if root_type == "workspace":
        base_path = _os.path.join(output_path, "workspace")
    else:
        base_path = output_path

    full_path = _os.path.normpath(_os.path.join(base_path, file_path))
    if not full_path.startswith(_os.path.normpath(base_path)):
        return jsonify({"code": 403, "message": "Path traversal denied"}), 403

    if not _os.path.isfile(full_path):
        return jsonify({"code": 404, "message": "File not found"}), 404

    try:
        with open(full_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception:
        with open(full_path, "rb") as f:
            content = f.read().decode("utf-8", errors="replace")

    ext = _os.path.splitext(full_path)[1].lower()
    from app.api.project import _ext_to_language
    return jsonify({
        "code": 200,
        "data": {
            "path": file_path,
            "content": content,
            "language": _ext_to_language(ext),
        }
    })
