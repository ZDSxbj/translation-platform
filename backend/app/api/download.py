import os
import zipfile
import io
from flask import Blueprint, jsonify, send_file, current_app
from app.services.pipeline_manager import get_pipeline_manager

download_bp = Blueprint("download", __name__)


@download_bp.route("/<session_id>/result", methods=["GET"])
def download_result(session_id):
    """Download the translated project as a ZIP file."""
    pipeline = get_pipeline_manager()
    state = pipeline.get_session_state(session_id)
    if state is None:
        return jsonify({"code": 404, "message": "Session not found"}), 404

    output_path = state.get("output_path", "")
    if not output_path or not os.path.isdir(output_path):
        return jsonify({"code": 404, "message": "Output not yet available"}), 404

    # Create in-memory ZIP
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(output_path):
            for fn in files:
                full_path = os.path.join(root, fn)
                arcname = os.path.relpath(full_path, output_path)
                zf.write(full_path, arcname)

    buf.seek(0)
    return send_file(
        buf,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"translated_{session_id[:8]}.zip",
    )


@download_bp.route("/<session_id>/report", methods=["GET"])
def download_report(session_id):
    """Download the translation report as JSON."""
    pipeline = get_pipeline_manager()
    state = pipeline.get_session_state(session_id)
    if state is None:
        return jsonify({"code": 404, "message": "Session not found"}), 404

    report = pipeline.get_report(session_id)
    if report is None:
        return jsonify({"code": 404, "message": "Report not yet available"}), 404

    buf = io.BytesIO()
    import json as _json
    report_bytes = _json.dumps(report, indent=2, ensure_ascii=False).encode("utf-8")
    buf.write(report_bytes)
    buf.seek(0)

    return send_file(
        buf,
        mimetype="application/json",
        as_attachment=True,
        download_name=f"report_{session_id[:8]}.json",
    )
