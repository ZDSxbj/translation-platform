# His2Trans Translation Platform — Backend

Flask-based REST API backend that orchestrates the His2Trans C→Rust translation pipeline.

---

## Architecture

```
HTTP Request (from frontend)
  → Blueprint (api/translate.py, api/upload.py, api/download.py)
    → PipelineManager (per-session state + stage gating)
      → His2TransEngine (engine.py)
        → FrameworkRunner (subprocess)
          → His2Trans framework scripts (in His2Trans-Opt-/framework/)
```

### Key Components

| Module | Path | Role |
|---|---|---|
| `translate_bp` | `app/api/translate.py` | Translation session lifecycle (create, run stages, get state, download) |
| `upload_bp` | `app/api/upload.py` | ZIP upload, extraction, file tree analysis |
| `download_bp` | `app/api/download.py` | Final project artifact download |
| `project_bp` | `app/api/project.py` | Project metadata and analysis |
| `PipelineManager` | `app/services/pipeline_manager.py` | Session management, stage state machine, report generation |
| `His2TransEngine` | `app/engines/his2trans/engine.py` | 4-stage pipeline orchestrator |
| `FrameworkRunner` | `app/engines/his2trans/runner.py` | Subprocess launcher with proper env + PYTHONPATH setup |
| `EnvMapper` | `app/engines/his2trans/env_mapper.py` | Maps platform config keys → framework env vars |
| `FileService` | `app/services/file_service.py` | Path resolution for uploads/outputs |
| `PathService` | `app/services/path_service.py` | Container-aware path classification |

### Engine Design

The `His2TransEngine` is a thin orchestrator (not a monolithic script wrapper). Each stage is a separate method:

- `_run_stage1()` — Runs `get_dependencies.py` → `skeleton_builder.py`. Detects `compile_commands.json`, auto-configures OHOS root for bindgen. Renames extracted output to match project name.
- `_run_stage2()` — Runs `generate_signature_mappings.py` → `run_jina_reranker_queued.py`. Symlinks RAG knowledge base + BM25 index into workspace.
- `_run_stage3()` — Runs `translate_function.py` → `auto_repair_rust.py` → `merge_final_project.py`. Populates signature match files, tracks compile results.
- `_run_postprocess()` — Aggregates results, generates downloadable package.

The engine calls framework scripts via `FrameworkRunner`, which:
1. Sets `C2R_WORKSPACE_ROOT` to the session's output workspace
2. Maps API keys (`API_KEY` → `EXTERNAL_API_KEY`) for the framework's LLM module
3. Sets `USE_VLLM=false` (framework defaults to local vLLM!)
4. Lowers `JINA_MIN_MEMORY_GB` to 4.0 GB (framework defaults to 8.0 GB)
5. Adds all framework subdirectories to `PYTHONPATH`
6. Streams stdout/stderr to the session log with timestamps

---

## API Endpoints

### Upload
| Method | Path | Description |
|---|---|---|
| `POST` | `/api/upload` | Upload `.zip` file, extract, return project tree + stats |

### Translation
| Method | Path | Description |
|---|---|---|
| `POST` | `/api/translate/start` | Create translation session |
| `GET` | `/api/translate/<id>/state` | Get session state + stage statuses |
| `POST` | `/api/translate/<id>/stage/<stage>/run` | Run a pipeline stage |
| `GET` | `/api/translate/<id>/stage/<stage>/result` | Get stage result |
| `GET` | `/api/translate/<id>/stage/<stage>/logs` | Get stage log entries |
| `GET` | `/api/translate/<id>/workspace/tree` | Browse workspace file tree |
| `GET` | `/api/translate/<id>/workspace/file` | Read a workspace file |
| `GET` | `/api/translate/<id>/report` | Get full translation report |

### Download
| Method | Path | Description |
|---|---|---|
| `GET` | `/api/download/<id>/final-project` | Download translated Rust project |
| `GET` | `/api/download/<id>/report` | Download report JSON |

### Project
| Method | Path | Description |
|---|---|---|
| `GET` | `/api/project/<id>/analyze` | Analyze project structure |
| `GET` | `/api/project/<id>/file` | Read source file contents |

---

## Configuration

Configuration is loaded from `config.py` + `.env`:

```python
# config.py
class Config:
    UPLOAD_FOLDER = "uploads"       # relative to backend/
    OUTPUT_FOLDER = "outputs"       # relative to backend/
    MAX_CONTENT_LENGTH = 200 * 1024 * 1024  # 200 MB max upload
    API_MODEL = "deepseek-v3.2"
```

Environment overrides (`.env`):
```
FLASK_ENV=development
API_KEY=your-api-key
API_BASE_URL=https://api.your-llm.com/v1
API_MODEL=deepseek-v3.2
HIS2TRANS_FRAMEWORK=/absolute/path/to/His2Trans-Opt-/framework
HIS2TRANS_DATA=/absolute/path/to/His2Trans-Opt-/data
```

---

## Testing

```bash
# Quick env mapper test (no framework needed)
python -m pytest tests/test_env_mapper.py -v

# Full integration test (requires His2Trans framework + LLM API)
# See tests/run_tests.sh for setup instructions
bash tests/run_tests.sh --full
```

---

## Resource Cleanup

The embedded framework copy at `app/engines/his2trans/framework/` has been trimmed from ~58 MB → 2.8 MB. Removed:
- `translation_outputs/` (46 MB of old test runs)
- `workspace/rag/` (8.5 MB — RAG data now self-contained as .tar.gz in `backend/data/rag/`)
- `data/nltk_data/` (duplicate — use `backend/data/nltk_data/` instead)
- `.cache/`, `mocks(notuse)/`, log files

The framework scripts themselves (`stage1_prep/`, `stage2_skeleton/`, `stage3_translate/`, `knowledge/`, `shared/`, `generate/`, `config/`) are kept as a fallback. At runtime, the engine uses the external framework path configured in `.env`.

### RAG Resources

The knowledge base (`knowledge_base.tar.gz`, 28 MB) and BM25 index (`bm25_index.tar.gz`, 2.6 MB) are shipped in `backend/data/rag/` as compressed archives. At startup, `His2TransEngine._ensure_rag_extracted()` auto-extracts them on first use — no external His2Trans-Opt- checkout needed for RAG. Extracted `.json`/`.pkl` files are gitignored.

---

## Known Issues

See the [root README](../README.md#known-limitations) for the full list. Backend-specific notes:

1. **Session storage is in-memory** — server restart loses all sessions. Use a database (Redis/SQLite) for production.
2. **No async stage execution** — stages run synchronously in the Flask request thread. For long stages (Stage 3 can take 10+ minutes), use a task queue (Celery/RQ).
3. **No WebSocket progress streaming** — SocketIO events are defined but not fully wired. Progress is polled via `/state` endpoint.
4. **Path fixup not automated** — If analysis detects `path_fixup_needed: true`, the engine should auto-call `relativize_paths()` but this is not yet wired.
