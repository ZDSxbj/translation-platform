"""Pipeline stage orchestrator with stage-by-stage user gating."""

import os
import threading
import time
import traceback
from datetime import datetime
from app.services.file_service import FileService
from app.services.engine_factory import get_engine

_pipeline_instance = None


def get_pipeline_manager():
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = PipelineManager()
    return _pipeline_instance


class PipelineSession:
    def __init__(self, session_id: str, project_id: str, config: dict,
                 upload_folder: str, output_folder: str):
        self.session_id = session_id
        self.project_id = project_id
        self.config = config
        # Ensure absolute paths — relative paths break subprocess CWD resolution
        self.upload_folder = os.path.abspath(upload_folder)
        self.output_folder = os.path.abspath(output_folder)
        self.source_path = FileService.get_source_dir(self.upload_folder, project_id)
        self.output_path = FileService.get_output_dir(self.output_folder, session_id)

        engine = get_engine(config["engine"])
        self.stages = []
        for s in engine.get_stages():
            self.stages.append({
                "id": s["id"],
                "name": s["name"],
                "status": "pending",  # pending, running, completed, failed, skipped
                "summary": "",
                "start_time": None,
                "end_time": None,
            })

        self.logs: dict[str, list[dict]] = {s["id"]: [] for s in self.stages}
        self.results: dict[str, dict] = {}
        self.current_stage_index = 0
        self.created_at = time.time()
        self._lock = threading.Lock()

    def add_log(self, stage_id: str, message: str, level: str = "info"):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "level": level,
        }
        with self._lock:
            if stage_id in self.logs:
                self.logs[stage_id].append(entry)

    def to_state_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "project_id": self.project_id,
            "config": self.config,
            "stages": self.stages,
            "current_stage_index": self.current_stage_index,
            "output_path": self.output_path,
        }


class PipelineManager:
    """Manages translation pipeline sessions."""

    def __init__(self):
        self.sessions: dict[str, PipelineSession] = {}

    def create_session(self, session_id: str, project_id: str, config: dict,
                       upload_folder: str, output_folder: str):
        session = PipelineSession(session_id, project_id, config, upload_folder, output_folder)
        self.sessions[session_id] = session

    def get_stages(self) -> list[dict]:
        """Get default stages from the engine."""
        return [
            {"id": "stage1_prep", "name": "Stage 1: Dependency Analysis + Skeleton", "status": "pending"},
            {"id": "stage2_rag", "name": "Stage 2: Signature Matching + RAG", "status": "pending"},
            {"id": "stage3_translate", "name": "Stage 3: Function Body Translation + Repair", "status": "pending"},
            {"id": "postprocess", "name": "Post-process: Reports & Packaging", "status": "pending"},
        ]

    def get_session_state(self, session_id: str) -> dict | None:
        session = self.sessions.get(session_id)
        if session is None:
            return None
        return session.to_state_dict()

    def run_stage(self, session_id: str, stage_id: str) -> dict:
        """Run a stage synchronously and return the result."""
        session = self.sessions.get(session_id)
        if session is None:
            raise ValueError(f"Session not found: {session_id}")

        # Find and update the stage
        stage = None
        for s in session.stages:
            if s["id"] == stage_id:
                stage = s
                break
        if stage is None:
            raise ValueError(f"Unknown stage: {stage_id}")

        if stage["status"] == "completed":
            raise ValueError(f"Stage already completed: {stage_id}")
        if stage["status"] == "running":
            # Frontend may have timed out and is retrying — reset and re-run
            session.add_log(stage_id, "Stage was marked running (likely timed-out frontend) — resetting and re-running", "warn")

        # Mark as running
        stage["status"] = "running"
        stage["start_time"] = datetime.now().isoformat()
        session.add_log(stage_id, f"Starting {stage['name']}...")

        try:
            engine = get_engine(session.config["engine"])

            def log_cb(msg, level="info"):
                session.add_log(stage_id, msg, level)

            # Prepare output directory
            FileService.ensure_dir(session.output_path)

            # Run the stage
            result = engine.run_stage(
                stage_id=stage_id,
                source_path=session.source_path,
                output_path=session.output_path,
                config=session.config,
                log_callback=log_cb,
            )

            stage["status"] = "completed"
            stage["end_time"] = datetime.now().isoformat()
            stage["summary"] = result.get("summary", "Completed")
            session.results[stage_id] = result

            # Advance current stage index
            for i, s in enumerate(session.stages):
                if s["id"] == stage_id and i == session.current_stage_index:
                    session.current_stage_index = i + 1
                    break

            session.add_log(stage_id, f"Stage completed: {stage['name']}")

            return {
                "stage_id": stage_id,
                "status": "completed",
                "summary": result.get("summary", ""),
                "details": result.get("details", {}),
            }

        except Exception as e:
            stage["status"] = "failed"
            stage["end_time"] = datetime.now().isoformat()
            stage["summary"] = str(e)[:200]
            session.add_log(stage_id, f"Error: {e}", "error")
            session.add_log(stage_id, traceback.format_exc(), "error")

            return {
                "stage_id": stage_id,
                "status": "failed",
                "summary": str(e)[:200],
                "details": {},
            }

    def get_stage_result(self, session_id: str, stage_id: str) -> dict | None:
        session = self.sessions.get(session_id)
        if session is None:
            return None
        return session.results.get(stage_id)

    def get_stage_logs(self, session_id: str, stage_id: str) -> list | None:
        session = self.sessions.get(session_id)
        if session is None:
            return None
        return session.logs.get(stage_id)

    def get_report(self, session_id: str) -> dict | None:
        session = self.sessions.get(session_id)
        if session is None:
            return None

        report = {
            "session_id": session_id,
            "project_id": session.project_id,
            "config": session.config,
            "stages": [],
            "generated_at": datetime.now().isoformat(),
        }

        # Collect pipeline statistics from workspace
        ws_stats = self._collect_workspace_stats(session.output_path)

        for stage in session.stages:
            entry = {
                "id": stage["id"],
                "name": stage["name"],
                "status": stage["status"],
                "summary": stage["summary"],
                "start_time": stage["start_time"],
                "end_time": stage["end_time"],
                "log_count": len(session.logs.get(stage["id"], [])),
            }
            if stage["id"] in session.results:
                entry["details"] = session.results[stage["id"]].get("details", {})
            report["stages"].append(entry)

        # Merge workspace stats into report for richer display
        if ws_stats:
            report.update(ws_stats)

        return report

    @staticmethod
    def _collect_workspace_stats(output_path: str) -> dict:
        """Collect statistics from workspace for the report."""
        stats = {}
        ws = os.path.join(output_path, "workspace")
        if not os.path.isdir(ws):
            return stats

        # Count Rust files in skeletons/
        skel_dir = os.path.join(ws, "skeletons")
        if os.path.isdir(skel_dir):
            rs_files = []
            for root, dirs, files in os.walk(skel_dir):
                for fn in files:
                    if fn.endswith(".rs"):
                        rs_files.append(os.path.relpath(os.path.join(root, fn), ws))
            stats["skeleton_rust_files"] = len(rs_files)

        # Count extracted functions
        for item in os.listdir(ws):
            if item.startswith("extracted") and os.path.isdir(os.path.join(ws, item)):
                manifest = os.path.join(ws, item, os.listdir(os.path.join(ws, item))[0],
                                       "functions_manifest.json") if os.listdir(os.path.join(ws, item)) else None
                if manifest and os.path.isfile(manifest):
                    try:
                        import json
                        with open(manifest) as f:
                            mf = json.load(f)
                        stats["extracted_functions"] = len(mf.get("functions", []))
                    except Exception:
                        pass
                break

        # Count translated functions
        trans_dir = os.path.join(ws, "translated")
        if os.path.isdir(trans_dir):
            total = 0
            for root, dirs, files in os.walk(trans_dir):
                total += sum(1 for f in files if f.endswith(".txt"))
            stats["translated_functions"] = total

        # Count signature matches (RAG results)
        sig_match_dir = os.path.join(ws, "signature_matches")
        if os.path.isdir(sig_match_dir):
            total = 0
            for root, dirs, files in os.walk(sig_match_dir):
                total += sum(1 for f in files if f.endswith(".txt"))
            stats["signature_matches"] = total

        # Count final merged project Rust files
        final_dir = os.path.join(ws, "final_projects")
        if os.path.isdir(final_dir):
            rs_count = 0
            for root, dirs, files in os.walk(final_dir):
                rs_count += sum(1 for f in files if f.endswith(".rs"))
            stats["final_rust_files"] = rs_count

        # Compile results: prefer incremental mode's translation_stats.json
        # over the legacy test_results/ directory (non-incremental mode).
        inc_stats = None
        inc_dir = os.path.join(ws, "incremental_work")
        if os.path.isdir(inc_dir):
            for item in os.listdir(inc_dir):
                ts_json = os.path.join(inc_dir, item, f"translate_by_{stats.get('model', '')}",
                                       "translation_stats.json")
                if not os.path.isfile(ts_json):
                    # Try with any model subdir
                    proj_dir = os.path.join(inc_dir, item)
                    if os.path.isdir(proj_dir):
                        for model_dir in os.listdir(proj_dir):
                            ts_json = os.path.join(proj_dir, model_dir, "translation_stats.json")
                            if os.path.isfile(ts_json):
                                break
                if os.path.isfile(ts_json):
                    try:
                        import json
                        with open(ts_json) as f:
                            inc_stats = json.load(f)
                    except Exception:
                        pass
                if inc_stats:
                    break

        if inc_stats:
            stats["compile_passed"] = inc_stats.get("compiled", 0)
            stats["compile_failed"] = inc_stats.get("failed", 0)
            stats["translated_functions"] = inc_stats.get("translated", 0)
            stats["repaired"] = inc_stats.get("repaired", 0)
            stats["c2rust_fallback"] = inc_stats.get("c2rust_fallback", 0)
        else:
            # Fallback: legacy test_results directory
            test_res_dir = os.path.join(ws, "test_results")
            if os.path.isdir(test_res_dir):
                passed = 0
                failed = 0
                for root, dirs, files in os.walk(test_res_dir):
                    for fn in files:
                        if fn.endswith(".txt"):
                            fpath = os.path.join(root, fn)
                            try:
                                with open(fpath, "r") as f:
                                    first_line = f.readline().strip()
                                if first_line.startswith("Success"):
                                    passed += 1
                                else:
                                    failed += 1
                            except Exception:
                                pass
                stats["compile_passed"] = passed
                stats["compile_failed"] = failed

        return stats
