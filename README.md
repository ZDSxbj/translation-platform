# His2Trans Translation Platform

A full-stack web platform for automated C/C++ → Rust code translation, built upon the His2Trans translation framework.

---

## Project Overview

**His2Trans Translation Platform** wraps the His2Trans research pipeline in a user-friendly web interface. Users upload C/C++ project archives (`.zip`), configure translation parameters, run a staged translation pipeline, inspect intermediate results at each stage, and download the final translated Rust project along with a comprehensive report.

### Core Pipeline (4 Stages)

| Stage | Name | What It Does |
|---|---|---|
| Stage 1 | Dependency Analysis + Skeleton | Extracts C functions, resolves include dependencies, runs `bindgen` to generate Rust type definitions and skeleton stubs |
| Stage 2 | Signature Matching + RAG | BM25 retrieval + Jina Reranker to match C functions against a knowledge base of previously translated Rust code |
| Stage 3 | Function Body Translation + Repair | LLM-based function body translation, `cargo check` compile validation, auto-repair loop |
| Post-process | Reports & Packaging | Aggregates statistics, generates downloadable report and final Rust project archive |

### Technology Stack

| Layer | Technology |
|---|---|
| **Frontend** | Vue 3 + Element Plus + Pinia + CodeMirror 6 |
| **Backend** | Flask (Python) + REST API |
| **Translation Engine** | His2Trans framework (Python) |
| **LLM Backend** | OpenAI-compatible API (DeepSeek, etc.) |
| **Code Parsing** | tree-sitter (C/C++, Rust grammars) |
| **RAG Retrieval** | BM25 + Jina Reranker (optional, GPU) — KB shipped as .tar.gz, auto-extracted |
| **Type Generation** | libclang + bindgen |

---

## Current Status

### Implemented

- [x] ZIP upload and project analysis
- [x] Full 4-stage pipeline orchestration with stage-by-stage gating
- [x] LLM translation with API key configuration
- [x] Auto-repair loop (compile-check → repair → re-check)
- [x] RAG signature matching (BM25 retrieval + Jina Reranker)
- [x] Intermediate file browsing per stage (skeleton .rs files, translated functions)
- [x] Final merged project download
- [x] Translation results report with statistics
- [x] OHOS (OpenHarmony) project support with `compile_commands.json` + bindgen
- [x] Standard C project support (no `compile_commands.json` required)
- [x] Workspace file tree viewer with CodeMirror code preview

### Known Limitations

- **Full project compilation**: Individual function compile-checks pass (26/26 on test OHOS project), but the merged final project may have residual compile errors from LLM-generated code (null byte literals, reserved keywords as identifiers, missing compat macros).
- **Standard C translation quality**: Without `compile_commands.json` (or a `bear`-generated one), bindgen has limited type context, producing stub types. Translation quality improves significantly with a compile_commands.json.
- **RAG BM25 search**: The BM25 retrieval step currently skips without results when the elastic_search directory is empty. Signature matching (C→Rust function signature mapping) works at 20/26 coverage.
- **GPU requirement for Jina Reranker**: RAG reranking requires a CUDA-capable GPU with ≥4GB free VRAM. Falls back to CPU but significantly slower.
- **No incremental/resume support**: If a stage fails, the entire stage must be re-run (partial progress within a stage is not preserved).
- **Single-project sessions**: Each session handles one project. No batch translation.

### Future Extensions

- [ ] Batch project translation queue
- [ ] Incremental translation resume (save/restore LLM progress)
- [ ] Multi-model comparison (run same project through different LLMs)
- [ ] Translation quality metrics (BLEU, CodeBLEU, AST diff)
- [ ] User authentication and project history
- [ ] Docker deployment with pre-configured GPU support
- [ ] GitHub integration (translate directly from repo URL)
- [ ] Plugin system for custom translation rules

---

## Quick Start

### Prerequisites

- **Python 3.10+** with pip
- **Node.js 18+** with npm
- **Conda** (recommended for environment isolation)
- **CUDA-capable GPU** (optional, for Jina Reranker RAG)
- **His2Trans-Opt- framework** (sibling directory at `../His2Trans-Opt-/framework`) — the framework scripts only; RAG knowledge base and BM25 index are self-contained in this repo as .tar.gz archives
- **Rust toolchain** (`rustc`, `cargo` — required for compile-check in Stage 3)

### 1. Backend Setup

```bash
cd backend

# Create and activate conda environment
conda create -n his2trans python=3.10 -y
conda activate his2trans

# Install Python dependencies
pip install -r requirements.txt

# Install tree-sitter language grammars (auto-compiled on first import)

# Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# Configure environment
cp .env.example .env
# Edit .env — set API_KEY, API_BASE_URL, paths to His2Trans-Opt-

# Start backend (port 5000)
FLASK_APP=run.py python -m flask run --host=0.0.0.0 --port=5000
```

### 2. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start dev server (port 8080, proxies API to :5000)
npm run dev
```

### 3. Open in Browser

Navigate to `http://localhost:8080`. Upload a C/C++ project ZIP, configure translation parameters, and step through the pipeline stages.

### Environment Variables (`.env`)

| Variable | Required | Description |
|---|---|---|
| `API_KEY` | Yes | LLM API key (OpenAI-compatible) |
| `API_BASE_URL` | Yes | LLM API base URL |
| `API_MODEL` | No | Model name (default: `deepseek-v3.2`) |
| `HIS2TRANS_FRAMEWORK` | Yes | Path to His2Trans-Opt-/framework |
| `HIS2TRANS_DATA` | Yes | Path to His2Trans-Opt-/data (OHOS SDK, not RAG) |
| `FLASK_ENV` | No | `development` or `production` |
| `PORT` | No | Backend port (default: 5000) |

See `backend/.env.example` for all available options.

---

## Project Structure

```
translation-platform/
├── backend/
│   ├── app/
│   │   ├── api/                  # REST API blueprints
│   │   │   ├── upload.py         # POST /api/upload
│   │   │   ├── translate.py      # POST /api/translate/* 
│   │   │   ├── download.py       # GET /api/download/*
│   │   │   └── project.py        # GET /api/project/*
│   │   ├── engines/
│   │   │   └── his2trans/        # His2Trans engine implementation
│   │   │       ├── engine.py     # Stage orchestrator
│   │   │       ├── runner.py     # Subprocess script runner
│   │   │       ├── env_mapper.py # Config → framework env var mapping
│   │   │       └── framework/    # Trimmed framework fallback (2.8 MB)
│   │   ├── services/             # Business logic
│   │   ├── socketio/             # Real-time progress events
│   │   └── utils/                # Zip utilities
│   ├── data/                     # Symlinked resources (gitignored)
│   ├── tests/                    # Integration tests
│   ├── config.py                 # App configuration
│   ├── run.py                    # Flask entry point
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── views/                # HomePage, TranslationWorkspace, ResultViewer
│   │   ├── components/           # Pipeline, repo, upload components
│   │   ├── stores/               # Pinia state management
│   │   ├── apis/                 # Axios API client
│   │   └── router/               # Vue Router
│   ├── vite.config.js
│   └── package.json
├── .gitignore
└── README.md
```
