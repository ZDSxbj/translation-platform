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

    engine = data.get("engine", "his2trans")

    # Fields common to all engines
    config = {
        "engine": engine,
        "ohos_root": data.get("ohos_root", ""),
        "extra_includes": data.get("extra_includes", []),
    }

    # His2Trans-specific fields — only include when relevant
    if engine != "c2rust":
        config.update({
            "model": data.get("model", current_app.config.get("API_MODEL", "deepseek-v3.2")),
            "use_rag": data.get("use_rag", False),
            "max_repair": int(data.get("max_repair", 5)),
            "api_key": data.get("api_key", current_app.config.get("API_KEY", "")),
            "api_base_url": data.get("api_base_url", current_app.config.get("API_BASE_URL", "")),
            "api_max_tokens": int(data.get("api_max_tokens", current_app.config.get("API_MAX_TOKENS", 8192))),
            "api_temperature": float(data.get("api_temperature", current_app.config.get("API_TEMPERATURE", 0.0))),
            "his2trans_framework": current_app.config.get("HIS2TRANS_FRAMEWORK", ""),
            "his2trans_data": current_app.config.get("HIS2TRANS_DATA", ""),
        })

    session_id = str(uuid.uuid4())
    pipeline = get_pipeline_manager()
    pipeline.create_session(
        session_id=session_id,
        project_id=project_id,
        config=config,
        upload_folder=current_app.config["UPLOAD_FOLDER"],
        output_folder=current_app.config["OUTPUT_FOLDER"],
    )

    # Return the session's *actual* stages (engine-dependent), not a
    # hardcoded list.  Different engines define different stage ids.
    session_state = pipeline.get_session_state(session_id)
    actual_stages = session_state["stages"] if session_state else pipeline.get_stages()

    return jsonify({
        "code": 200,
        "message": "Translation session created",
        "data": {
            "session_id": session_id,
            "stages": [{"id": s["id"], "name": s["name"], "status": s["status"]}
                       for s in actual_stages],
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

    # Filter out build artifacts for final_projects display
    def _filter_artifacts(nodes):
        _skip_dirs = {"target", "native", ".c2r_bindgen_extern", ".c2r_c2rust_fallback",
                      "__c2r_generated"}
        _skip_exts = {".o", ".bin", ".d", ".rmeta", ".rlib", ".a",
                      ".json", ".log", ".txt", ".lock", ".timestamp"}
        _skip_names = {"build_result.txt", "compile_error_full.log",
                       "translation_stats.json", "types_generation_report.json",
                       "types_recovery_report.json", "CACHEDIR.TAG",
                       ".rustc_info.json"}
        result = []
        for node in nodes:
            if node.get("name", "") in _skip_dirs or node.get("name", "") in _skip_names:
                continue
            ext = _os.path.splitext(node.get("name", ""))[1].lower()
            if node["type"] == "file" and ext in _skip_exts:
                continue
            if "children" in node and node["children"] is not None:
                node["children"] = _filter_artifacts(node["children"])
            result.append(node)
        return result

    tree = _filter_artifacts(tree)
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


@translate_bp.route("/<session_id>/rag/knowledge", methods=["GET"])
def get_rag_knowledge(session_id):
    """Get RAG-matched knowledge snippets for the session's project."""
    import os as _os

    pipeline = get_pipeline_manager()
    session = pipeline.sessions.get(session_id)
    if session is None:
        return jsonify({"code": 404, "message": "Session not found"}), 404

    workspace = _os.path.join(session.output_path, "workspace")
    knowledge = []
    for rag_subdir in ("reranked_results", "elastic_search_results"):
        rag_dir = _os.path.join(workspace, "rag", rag_subdir)
        if not _os.path.isdir(rag_dir):
            continue
        for proj_dir in _os.listdir(rag_dir):
            proj_path = _os.path.join(rag_dir, proj_dir)
            if not _os.path.isdir(proj_path):
                continue
            for fn in sorted(_os.listdir(proj_path)):
                if not fn.endswith(".txt"):
                    continue
                fpath = _os.path.join(proj_path, fn)
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read().strip()
                except Exception:
                    continue
                if not content:
                    continue
                entry = {"func_file": fn.replace(".txt", ""), "raw": content[:3000]}
                current_key = None
                current_body = []
                for line in content.split("\n"):
                    if line.startswith("C_Code:") or line.startswith("C_Code "):
                        if current_key and current_body:
                            val = "\n".join(current_body).strip()[:500]
                            if current_key == "c_code":
                                entry["c_code"] = val
                            elif current_key == "rust_code":
                                entry["rust_code"] = val
                        current_key = "c_code"
                        current_body = [line.split(":", 1)[-1].strip()] if ":" in line else []
                    elif line.startswith("Function:") or line.startswith("Function "):
                        if current_key == "c_code" and current_body:
                            entry["c_code"] = "\n".join(current_body).strip()[:500]
                        current_key = "rust_code"
                        current_body = [line.split(":", 1)[-1].strip()] if ":" in line else []
                    elif line.startswith("---") or line.startswith("Unixcoder"):
                        if current_key and current_body:
                            val = "\n".join(current_body).strip()[:500]
                            if current_key == "c_code":
                                entry["c_code"] = val
                            elif current_key == "rust_code":
                                entry["rust_code"] = val
                        current_key = None
                        current_body = []
                    elif current_key:
                        current_body.append(line)
                if current_key and current_body:
                    val = "\n".join(current_body).strip()[:500]
                    if current_key == "c_code":
                        entry["c_code"] = val
                    elif current_key == "rust_code":
                        entry["rust_code"] = val
                if entry.get("c_code") or entry.get("rust_code"):
                    knowledge.append(entry)
        if knowledge:
            break

    return jsonify({"code": 200, "data": {"knowledge": knowledge, "count": len(knowledge)}})


@translate_bp.route("/<session_id>/functions/comparison", methods=["GET"])
def get_function_comparison(session_id):
    """Return C/Rust side-by-side comparison data for each translated function."""
    import os as _os, json as _json

    pipeline = get_pipeline_manager()
    session = pipeline.sessions.get(session_id)
    if session is None:
        return jsonify({"code": 404, "message": "Session not found"}), 404

    ws = _os.path.join(session.output_path, "workspace")
    functions = []

    # --- Read C source from uploaded project ---
    c_funcs: dict[str, str] = {}   # func_file → c_code
    # Read functions_manifest.json to map func_file → source_file
    manifest_path = None
    extr_root = _os.path.join(ws, "extracted")
    if _os.path.isdir(extr_root):
        for item in _os.listdir(extr_root):
            mf = _os.path.join(extr_root, item, "functions_manifest.json")
            if _os.path.isfile(mf):
                manifest_path = mf
                break

    func_meta: dict[str, dict] = {}  # func_file → {name, source_file, start_line, end_line}
    if manifest_path:
        try:
            mf_data = _json.load(open(manifest_path, encoding="utf-8"))
            for f in mf_data.get("functions", []):
                ff = f.get("func_file", "")
                if ff and ff.endswith(".txt"):
                    ff = ff[:-4]
                func_meta[ff] = {
                    "name": f.get("name", ""),
                    "source_file": f.get("source_file", ""),
                    "start_line": f.get("start_line", 0),
                    "end_line": f.get("end_line", 0),
                }
        except Exception:
            pass

    # Read C code from uploaded source — search by function name
    source_base = session.source_path
    source_cache: dict[str, str] = {}
    _re_cfunc = __import__("re").compile(r'')  # placeholder, built per-function
    for ff, meta in func_meta.items():
        src_file = meta.get("source_file", "")
        fn_name = meta.get("name", "")
        c_path = _os.path.join(source_base, src_file)
        if not _os.path.isfile(c_path) or not fn_name:
            continue
        if src_file not in source_cache:
            try:
                source_cache[src_file] = open(c_path, encoding="utf-8", errors="ignore").read()
            except Exception:
                source_cache[src_file] = ""
        full = source_cache[src_file]
        if not full:
            continue
        try:
            pat = __import__("re").compile(
                __import__("re").escape(fn_name) + r'\s*\([^)]*\)\s*\n?\s*\{')
            m = pat.search(full)
            if m:
                start = m.start()
                brace_i = full.index('{', m.end() - 1)
                depth = 0
                for j in range(brace_i, len(full)):
                    if full[j] == '{': depth += 1
                    elif full[j] == '}':
                        depth -= 1
                        if depth == 0:
                            c_funcs[ff] = full[start:j+1]
                            break
                if ff not in c_funcs:
                    c_funcs[ff] = full[start:start+500]
            else:
                c_funcs[ff] = ""
        except Exception:
            c_funcs[ff] = ""

    # --- Read Rust output: try final_projects, then incremental_work, then skeletons ---
    rust_files: dict[str, str] = {}  # src_hdf_device_info.rs → content
    for base_dir in ("final_projects", "incremental_work", "skeletons"):
        base = _os.path.join(ws, base_dir)
        if not _os.path.isdir(base):
            continue
        for proj in _os.listdir(base):
            proj_dir = _os.path.join(base, proj)
            if not _os.path.isdir(proj_dir):
                continue
            src_dir = None
            # Try <proj>/<model>/src/ or <proj>/src/ directly
            for sub in _os.listdir(proj_dir):
                sub_dir = _os.path.join(proj_dir, sub, "src")
                if _os.path.isdir(sub_dir):
                    src_dir = sub_dir
                    break
            if not src_dir:
                src_dir = _os.path.join(proj_dir, "src")
            if not _os.path.isdir(src_dir):
                continue
            for rs_fn in _os.listdir(src_dir):
                if rs_fn.endswith(".rs") and (rs_fn.startswith("src_") or rs_fn == "main.rs"):
                    if rs_fn not in rust_files:
                        try:
                            rust_files[rs_fn] = open(
                                _os.path.join(src_dir, rs_fn),
                                encoding="utf-8", errors="ignore").read()
                        except Exception:
                            pass
            if rust_files:
                break
        if rust_files:
            break

    # --- load incremental stats (search all subdirs) ---
    inc_stats: dict[str, dict] = {}
    for root, _dirs, files in _os.walk(_os.path.join(ws, "incremental_work")):
        for fn in files:
            if fn == "translation_stats.json":
                try:
                    inc_stats = _json.load(open(_os.path.join(root, fn), encoding="utf-8"))
                except Exception:
                    pass
                break
        if inc_stats:
            break

    # --- map func_file to .rs file ---
    def _infer_rs(ffn):
        base = ffn[:-4] if ffn.endswith(".txt") else ffn
        parts = base.split("_")
        if len(parts) > 1 and parts[-1].isdigit():
            return "_".join(parts[:-1]) + ".rs"
        for i, p in enumerate(parts):
            if i > 0 and p and p[0].isupper():
                return "_".join(parts[:i]) + ".rs"
        if len(parts) > 1:
            return "_".join(parts[:-1]) + ".rs"
        return base + ".rs"

    compiled_count = inc_stats.get("compiled", 0)
    failed_count = inc_stats.get("failed", 0)
    repaired_count = inc_stats.get("repaired", 0)

    # --- helper: extract a single function body from a .rs file ---
    def _extract_rust_fn(content: str, func_name: str) -> str:
        pattern = (
            r'(?:pub\s+(?:extern\s+"C"\s+)?)?fn\s+'
            + _re_mod.escape(func_name)
            + r'\s*\([^)]*\)[^{]*\{'
        )
        m = _re_mod.search(pattern, content)
        if not m:
            return ""
        start = m.start()
        depth = 0
        i = m.end() - 1  # position of '{'
        while i < len(content):
            if content[i] == '{':
                depth += 1
            elif content[i] == '}':
                depth -= 1
                if depth == 0:
                    return content[start : i + 1]
            i += 1
        return ""

    # --- per-function source detection ---
    # Parse Stage 3 session logs to count repair rounds per function.
    # Each 位置索引注入成功 marks an injection; repairs show additional
    # injections after [Repair统计].
    func_injections: dict[str, int] = {}  # func_name → injection count
    stage3_logs = pipeline.get_stage_logs(session_id, "stage3_translate") or []
    for entry in stage3_logs:
        msg = entry.get("message", "")
        if "位置索引注入成功:" in msg:
            fn = msg.split("位置索引注入成功:")[-1].strip()
            func_injections[fn] = func_injections.get(fn, 0) + 1

    # Check which functions were C2Rust-fallback (calls __c2rust_fallback module)
    func_c2rust: set[str] = set()      # func_name → is C2Rust
    func_failed: set[str] = set()      # func_name → truly failed (impossible with 26/26)
    # Find functions whose body delegates to a C2Rust fallback wrapper.
    # The C2Rust module contains ALL functions from the file group, but
    # only those with `crate::compat::__c2rust_fallback::` in their body
    # were actually filled by C2Rust fallback.
    _re_mod = __import__("re")
    for _root, _dirs, _files in _os.walk(ws):
        for _fn in _files:
            if _fn.endswith(".rs") and _fn.startswith("src_") and not _fn.endswith("types.rs"):
                try:
                    _content = open(_os.path.join(_root, _fn), encoding="utf-8", errors="ignore").read()
                    # For each __c2rust_fallback occurrence, find the nearest
                    # preceding function signature (the one whose body
                    # actually contains the fallback call).
                    for _c2r in _re_mod.finditer(r'__c2rust_fallback', _content):
                        _before = _content[:_c2r.start()]
                        _fn_matches = list(_re_mod.finditer(
                            r'(?:pub\s+)?(?:unsafe\s+)?(?:extern\s+"C"\s+)?fn\s+(\w+)\s*\([^)]*\)',
                            _before))
                        if _fn_matches:
                            func_c2rust.add(_fn_matches[-1].group(1))
                except Exception:
                    pass

    for func_file, c_code in sorted(c_funcs.items()):
        rs_file = _infer_rs(func_file)
        fn_name = func_meta.get(func_file, {}).get("name", "")
        full_file = rust_files.get(rs_file, "")
        rust_code = _extract_rust_fn(full_file, fn_name) if full_file and fn_name else ""

        # Determine source and repair rounds from injection count
        injections = func_injections.get(fn_name, 0)
        repair_rounds = max(0, injections - 1)
        is_c2rust = fn_name in func_c2rust

        if is_c2rust:
            source_by = "c2rust"
            status_tag = "🤖 C2Rust"
        elif repair_rounds > 0:
            source_by = "llm_repaired"
            status_tag = f"🔧 LLM ({repair_rounds} rounds)"
        else:
            source_by = "llm_one_shot"
            status_tag = "✅ LLM (1-shot)"

        functions.append({
            "func_file": func_file,
            "func_name": fn_name,
            "c_code": c_code[:3000],
            "rust_code": rust_code,
            "rust_file": rs_file,
            "source_by": source_by,
            "repair_rounds": repair_rounds,
            "status_tag": status_tag,
        })

    return jsonify({
        "code": 200,
        "data": {
            "functions": functions,
            "compiled": compiled_count,
            "repaired": repaired_count,
            "failed": failed_count,
        }
    })


@translate_bp.route("/<session_id>/stage/stage1_prep/visualization", methods=["GET"])
def get_stage1_visualization(session_id):
    """Return Stage 1 visualization data: skeleton file list + call graph."""
    import os as _os, json as _json, re as _re

    pipeline = get_pipeline_manager()
    session = pipeline.sessions.get(session_id)
    if session is None:
        return jsonify({"code": 404, "message": "Session not found"}), 404

    ws = _os.path.join(session.output_path, "workspace")
    source_base = session.source_path

    # --- skeleton file list ---
    skeleton_files = []
    skel_dirs = []
    skel_root = _os.path.join(ws, "skeletons")
    if _os.path.isdir(skel_root):
        skel_dirs = [_os.path.join(skel_root, d) for d in _os.listdir(skel_root)
                     if _os.path.isdir(_os.path.join(skel_root, d))]

    opaque_types = []
    for skel_dir in skel_dirs:
        src_dir = _os.path.join(skel_dir, "src")
        if not _os.path.isdir(src_dir):
            continue
        for fn in sorted(_os.listdir(src_dir)):
            if not fn.endswith(".rs"):
                continue
            fpath = _os.path.join(src_dir, fn)
            try:
                content = open(fpath, encoding="utf-8", errors="ignore").read()
            except Exception:
                content = ""
            lines = content.split("\n")
            fn_count = sum(1 for l in lines if _re.search(
                r'(?:pub\s+)?(?:unsafe\s+)?(?:extern\s+"C"\s+)?fn\s+\w+', l))
            # Count structs/opaque types
            struct_count = len(_re.findall(r'pub struct (\w+)', content))
            opaque_count = len(_re.findall(r'pub struct \w+\s*\{\s*_(?:private|opaque|unused)', content))
            if opaque_count > 0:
                opaque_types.extend(_re.findall(
                    r'pub struct (\w+)\s*\{\s*_(?:private|opaque|unused)', content))
            # Build full workspace-relative path for frontend file loading
            _proj_dir_name = _os.path.basename(skel_dir)
            skeleton_files.append({
                "name": fn, "path": f"skeletons/{_proj_dir_name}/src/{fn}",
                "fn_count": fn_count, "struct_count": struct_count,
                "opaque_count": opaque_count, "size": len(lines),
            })

    # --- call graph ---
    call_graph = {"nodes": [], "edges": []}
    nodes_seen = set()
    cg_path = None
    extr_root = _os.path.join(ws, "extracted")
    if _os.path.isdir(extr_root):
        for item in _os.listdir(extr_root):
            cg = _os.path.join(extr_root, item, "call_graph.json")
            if _os.path.isfile(cg):
                cg_path = cg
                break
    # Build func_meta mapping for call graph: func_name → {source_file}
    func_meta_viz: dict[str, dict] = {}

    # Get project function names from manifest (needed for call graph filtering)
    __proj_names = set()
    _mf_path = None
    extr_root2 = _os.path.join(ws, "extracted")
    if _os.path.isdir(extr_root2):
        for item in _os.listdir(extr_root2):
            mf = _os.path.join(extr_root2, item, "functions_manifest.json")
            if _os.path.isfile(mf):
                _mf_path = mf
                break
    if _mf_path:
        try:
            _mf_data = _json.load(open(_mf_path, encoding="utf-8"))
            for f in _mf_data.get("functions", []):
                fn = f.get("name", "")
                if fn:
                    func_meta_viz[fn] = {"name": fn, "source_file": f.get("source_file", "")}
            __proj_names = {f.get("name", "") for f in _mf_data.get("functions", []) if f.get("name")}
        except Exception:
            pass

    if cg_path and __proj_names:
        try:
            cg_data = _json.load(open(cg_path, encoding="utf-8"))
            funcs = cg_data.get("functions", {})
            cg_edges = cg_data.get("call_graph", {})
            name_idx = cg_data.get("name_index", {})

            # Project function names (already loaded from manifest above)
            seen_ids = set()

            for func_name in sorted(__proj_names):
                func_ids = name_idx.get(func_name, [])
                if isinstance(func_ids, str):
                    func_ids = [func_ids]
                if not func_ids:
                    continue
                # Use function name as node id (unique for project functions)
                if func_name in seen_ids:
                    continue
                seen_ids.add(func_name)
                call_graph["nodes"].append({"id": func_name, "label": func_name, "is_external": False})
                # Add callee edges from first matching func_id
                for fid in func_ids:
                    for tgt_id in (cg_edges.get(fid) if isinstance(cg_edges, dict) else []) or []:
                        tgt_info = funcs.get(tgt_id, {})
                        tgt_name = ""
                        if isinstance(tgt_info, dict):
                            tgt_name = tgt_info.get("name", "") or ""
                        if not tgt_name:
                            tgt_name = tgt_id.split(":")[-1] if ":" in tgt_id else tgt_id
                        is_ext = tgt_name not in __proj_names
                        if tgt_name not in seen_ids:
                            seen_ids.add(tgt_name)
                            call_graph["nodes"].append({"id": tgt_name, "label": tgt_name, "is_external": is_ext})
                        call_graph["edges"].append({"source": func_name, "target": tgt_name, "id": f"{func_name}__{tgt_name}", "is_external": is_ext})
        except Exception:
            pass

    # --- preprocessing stats ---
    preprocess_files = []
    pp_dir = _os.path.join(ws, ".preprocessed")
    if _os.path.isdir(pp_dir):
        for fn in sorted(_os.listdir(pp_dir)):
            if fn.endswith(".i"):
                fpath = _os.path.join(pp_dir, fn)
                try:
                    content = open(fpath, encoding="utf-8", errors="ignore").read()
                except Exception:
                    content = ""
                preprocess_files.append({
                    "name": fn,
                    "size_lines": len(content.split("\n")),
                })

    return jsonify({"code": 200, "data": {
        "skeleton_files": skeleton_files,
        "call_graph": call_graph,
        "preprocess_files": preprocess_files,
        "opaque_types": list(set(opaque_types)),
    }})


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
