"""WebSocket events for real-time log streaming during pipeline execution.

This module provides the Socket.IO event handlers. The socketio server
is initialized in app/__init__.py when flask-socketio is available.
"""

# Socket.IO events:
# - Client connects to namespace '/pipeline'
# - Client joins room: session_id
# - Server emits to room:
#     'log'        — { stage_id, message, level, timestamp }
#     'progress'   — { stage_id, percent, message }
#     'stage_complete' — { stage_id, result_summary }
#     'stage_error' — { stage_id, error_message }

try:
    from flask_socketio import SocketIO, emit, join_room, leave_room

    socketio = SocketIO(cors_allowed_origins="*", async_mode="threading")

    @socketio.on("join", namespace="/pipeline")
    def on_join(data):
        session_id = data.get("session_id")
        if session_id:
            join_room(session_id)
            emit("joined", {"session_id": session_id, "status": "ok"})

    @socketio.on("leave", namespace="/pipeline")
    def on_leave(data):
        session_id = data.get("session_id")
        if session_id:
            leave_room(session_id)
            emit("left", {"session_id": session_id, "status": "ok"})

    def emit_log(session_id: str, stage_id: str, message: str, level: str = "info"):
        """Emit a log message to all clients in the session room."""
        from datetime import datetime
        socketio.emit("log", {
            "stage_id": stage_id,
            "message": message,
            "level": level,
            "timestamp": datetime.now().isoformat(),
        }, room=session_id, namespace="/pipeline")

    def emit_progress(session_id: str, stage_id: str, percent: float, message: str = ""):
        socketio.emit("progress", {
            "stage_id": stage_id,
            "percent": percent,
            "message": message,
        }, room=session_id, namespace="/pipeline")

    def emit_stage_complete(session_id: str, stage_id: str, summary: str):
        socketio.emit("stage_complete", {
            "stage_id": stage_id,
            "summary": summary,
        }, room=session_id, namespace="/pipeline")

    def emit_stage_error(session_id: str, stage_id: str, error: str):
        socketio.emit("stage_error", {
            "stage_id": stage_id,
            "error": error,
        }, room=session_id, namespace="/pipeline")

except ImportError:
    # flask-socketio not installed — no-op stubs
    socketio = None

    def emit_log(*args, **kwargs): pass
    def emit_progress(*args, **kwargs): pass
    def emit_stage_complete(*args, **kwargs): pass
    def emit_stage_error(*args, **kwargs): pass
