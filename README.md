# His2Trans Translation Platform

A full-stack web platform for automated C/C++ → Rust code translation, built upon the His2Trans research framework.

---

## Project Overview

**His2Trans Translation Platform** wraps the His2Trans research pipeline in a user-friendly web interface. Users upload C/C++ project archives (`.zip`), configure translation parameters, run a staged translation pipeline, inspect intermediate results at each stage, and download the final translated Rust project along with a comprehensive report.

### Core Pipeline (4 Stages)

| Stage | Name | What It Does |
|---|---|---|
| Stage 1 | Dependency Analysis + Skeleton | Extracts C functions via tree-sitter, resolves include dependencies via `clang -E` preprocessing, runs `bindgen` to generate Rust type definitions and skeleton stubs |
| Stage 2 | Signature Matching + RAG | BM25 retrieval + Jina Reranker to match C functions against a knowledge base of previously translated Rust code |
| Stage 3 | Function Body Translation + Repair | Incremental LLM translation (function-by-function), `cargo check` compile validation, auto-repair loop, C2Rust deterministic fallback, compat.rs auto-fill — all integrated in `incremental_translate.py` |
| Post-process | Reports & Packaging | Aggregates statistics, generates downloadable report and final Rust project archive |

### Technology Stack

| Layer | Technology |
|---|---|
| **Frontend** | Vue 3 + Element Plus + Pinia + CodeMirror 6 |
| **Backend** | Flask (Python) + REST API |
| **Translation Engine** | His2Trans framework (bundled in-tree, ~20K lines) |
| **LLM Backend** | OpenAI-compatible API (Claude Opus 4.8, DeepSeek v3.2, Qwen3-Coder, etc.) |
| **Code Parsing** | tree-sitter (C/C++, Rust grammars) |
| **RAG Retrieval** | BM25 + Jina Reranker (optional, GPU) — KB shipped as .tar.gz, auto-extracted |
| **Type Generation** | clang + bindgen |
| **Fallback Transpiler** | C2Rust (optional, `cargo install c2rust`) |

---

## Current Status

### Implemented

- [x] ZIP upload and project analysis
- [x] Full 4-stage pipeline orchestration with stage-by-stage gating
- [x] LLM translation with API key configuration (OpenAI-compatible: Claude, DeepSeek, Qwen, GPT, etc.)
- [x] Incremental translation with per-function compile verification and auto-repair
- [x] C2Rust deterministic fallback for functions that fail all LLM repair attempts
- [x] Compat.rs auto-fill via TU preprocessing (Step 2.55 extern declaration injection)
- [x] RAG signature matching (BM25 retrieval + Jina Reranker)
- [x] Intermediate file browsing per stage (skeleton .rs files, translated functions)
- [x] Final merged project download
- [x] Translation results report with real statistics (compile pass/fail, repair count, C2Rust fallback count)
- [x] OHOS (OpenHarmony) project support with `compile_commands.json` + bindgen
- [x] Standard C project support (no `compile_commands.json` required)
- [x] Workspace file tree viewer with CodeMirror code preview

### Translation Quality (Tested: OHOS HDF "shared" module, 26 functions)

| Model | One-Shot | After Repair | C2Rust Fallback | Total |
|---|---|---|---|---|
| Claude Opus 4.8 | 22 (85%) | 4 (15%) | 0 | **26/26 (100%)** |
| DeepSeek v3.2 | 9 (35%) | 14 (54%) | 3 (11%) | **26/26 (100%)** |

Final merged project compiles successfully (`cargo check` passes).

### Known Limitations

- **Jina Reranker requires GPU**: RAG reranking needs a CUDA-capable GPU with ≥4GB free VRAM. Pre-computed reranked results are included in the repo as a fallback for the OHOS test projects.
- **ohos_root_min is minimal**: The bundled OHOS header subset may miss definitions for some macros/functions. Add missing headers to `backend/data/ohos/ohos_root_min/` and rebuild the archive as needed.
- **C2Rust optional**: The C2Rust fallback requires `cargo install c2rust` and `cmake`. Without it, functions that fail all LLM repair attempts will be left as `unimplemented!()` stubs (final project still compiles).
- **Single-project sessions**: Each session handles one project. No batch translation.
- **In-memory session storage**: Sessions are lost on server restart (no database persistence).

### Future Extensions

- [ ] Batch project translation queue
- [ ] Incremental translation resume (save/restore LLM progress)
- [ ] Multi-model comparison (run same project through different LLMs)
- [ ] Translation quality metrics (BLEU, CodeBLEU, AST diff)
- [ ] User authentication and project history
- [ ] Docker deployment with pre-configured GPU support
- [ ] GitHub integration (translate directly from repo URL)

---

## Quick Start

### Prerequisites

| Tool | Required | Install |
|---|---|---|
| Python 3.10+ | Yes | system / conda |
| Node.js 18+ | Yes | system |
| Rust toolchain (nightly recommended) | Yes | `rustup toolchain install nightly` |
| clang + libclang | Yes (preprocessing + bindgen) | `sudo apt install clang libclang-dev` |
| cmake | Optional (C2Rust fallback) | `sudo apt install cmake` |
| C2Rust | Optional (deterministic fallback) | `cargo install c2rust` |
| CUDA GPU | Optional (Jina Reranker) | hardware |

**Note**: The His2Trans framework is bundled in-tree at `backend/app/engines/his2trans/framework/` — no external framework repository is required. The `.env` placeholder path (`/absolute/path/to/...`) is automatically detected and falls back to the bundled copy.

### 1. Backend Setup

```bash
cd backend

# Install Python dependencies
pip install -r requirements.txt

# Download NLTK data (one-time)
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# Configure environment
cp .env.example .env
# Edit .env — set API_KEY, API_BASE_URL, API_MODEL

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
| `API_MODEL` | No | Model name (default: `claude-opus-4-8`) |
| `API_MAX_TOKENS` | No | Max tokens per request (default: 8192) |
| `API_TEMPERATURE` | No | Sampling temperature (default: 0.0) |
| `API_TIMEOUT` | No | Request timeout in seconds (default: 600) |
| `HIS2TRANS_FRAMEWORK` | No | Override bundled framework path (auto-detected if not set) |
| `FLASK_ENV` | No | `development` or `production` |
| `CORS_ORIGINS` | No | CORS allowed origins (default: `*`) |
| `VLLM_MAX_RETRIES` | No | LLM request retries (default: 3) |

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
│   │   │       ├── engine.py     # Stage orchestrator (~600 lines)
│   │   │       ├── runner.py     # Subprocess script runner
│   │   │       ├── env_mapper.py # Config → framework env var mapping
│   │   │       └── framework/    # Bundled framework (in-tree, ~20K lines)
│   │   │           ├── stage1_prep/     # Dependency analysis + preprocessing
│   │   │           ├── stage2_skeleton/ # bindgen types + skeleton builder
│   │   │           ├── stage3_translate/# incremental_translate.py + C2Rust
│   │   │           ├── knowledge/       # BM25 + Jina Reranker
│   │   │           └── generate/        # LLM generation module
│   │   ├── services/             # Business logic
│   │   │   ├── pipeline_manager.py
│   │   │   ├── path_service.py
│   │   │   └── file_service.py
│   │   └── utils/
│   ├── data/
│   │   ├── ohos/
│   │   │   ├── ohos_root_min/      # Minimal OHOS header tree
│   │   │   └── ohos_root_min.tar.gz # (auto-extracted on first use)
│   │   └── rag/                    # Knowledge base + BM25 index
│   ├── config.py                 # App configuration
│   ├── run.py                    # Flask entry point
│   ├── .env                      # Local configuration (gitignored)
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
