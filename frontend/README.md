# His2Trans Translation Platform — Frontend

Vue 3 + Element Plus SPA for the His2Trans C→Rust translation platform.

---

## Tech Stack

| Category | Library |
|---|---|
| Framework | Vue 3 (Composition API, `<script setup>`) |
| State Management | Pinia |
| Routing | Vue Router 4 |
| UI Components | Element Plus + @element-plus/icons-vue |
| HTTP Client | Axios |
| Code Editor | CodeMirror 6 (C/C++ and Rust language support) |
| Build Tool | Vite 5 |

---

## Project Structure

```
src/
├── views/                        # Page-level components (routed)
│   ├── HomePage.vue              # Upload + analysis entry point
│   ├── TranslationWorkspace.vue  # Stage-by-stage pipeline UI
│   └── ResultViewer.vue          # Final report + download
├── components/
│   ├── upload/ZipUploader.vue    # Drag-and-drop ZIP upload
│   ├── config/
│   │   ├── AnalysisPanel.vue     # Project tree + stats after upload
│   │   └── ConfigPanel.vue       # LLM config + pipeline options
│   ├── pipeline/
│   │   ├── PipelineStepper.vue   # Stage progress stepper
│   │   ├── StagePanel.vue        # Per-stage output (file tree + code)
│   │   └── StageLogViewer.vue    # Real-time log viewer
│   ├── repo/
│   │   ├── FileTree.vue          # Recursive file tree
│   │   ├── FileTreeNode.vue      # Single tree node
│   │   └── CodeViewer.vue        # CodeMirror code display
│   ├── layout/AppHeader.vue      # Top navigation bar
│   └── common/
│       ├── LoadingOverlay.vue    # Full-page loading spinner
│       └── StatusBadge.vue       # Colored stage status badge
├── stores/translation.js         # Pinia store (session, stages, logs)
├── apis/index.js                 # Axios API client
├── router/index.js               # Vue Router config
├── assets/styles/main.css        # Global styles
├── App.vue                       # Root component
└── main.js                       # App entry point
```

---

## Key Views

### 1. HomePage (`/`)
Upload a C/C++ project `.zip` file. After upload, the project file tree is displayed with statistics. User configures translation parameters (LLM model, RAG toggle, max repair rounds) and clicks "Start Translation".

### 2. TranslationWorkspace (`/translate/:sessionId`)
The main pipeline workspace. Shows stage cards in a vertical stepper:
- Completed stages show their output (file tree + code viewer)
- The current stage shows a "Run" button
- After each stage completes, the next stage becomes available
- Each stage card shows logs, intermediate file tree, and code preview

### 3. ResultViewer (`/result/:sessionId`)
Final results page:
- Pipeline summary with per-stage status
- Statistics dashboard (extracted functions, translated functions, compile pass/fail)
- Download buttons for the final Rust project and report

---

## API Integration

The Axios client (`src/apis/index.js`) maps all backend endpoints:

```js
// Upload
uploadZip(formData)          → POST /api/upload

// Translation
startTranslation(config)      → POST /api/translate/start
getSessionState(sessionId)    → GET /api/translate/<id>/state
runStage(sessionId, stageId)  → POST /api/translate/<id>/stage/<stage>/run
getStageResult(sessionId, id) → GET /api/translate/<id>/stage/<stage>/result
getStageLogs(sessionId, id)   → GET /api/translate/<id>/stage/<stage>/logs
getWorkspaceTree(sessionId)   → GET /api/translate/<id>/workspace/tree
getWorkspaceFile(sessionId)   → GET /api/translate/<id>/workspace/file
getReport(sessionId)          → GET /api/translate/<id>/report

// Download
downloadProject(sessionId)    → GET /api/download/<id>/final-project
downloadReport(sessionId)     → GET /api/download/<id>/report
```

Vite dev server proxies `/api/*` requests to `http://localhost:5000` (configured in `vite.config.js`).

---

## Development

```bash
# Install dependencies
npm install

# Start dev server (port 8080)
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

### Configuration

`vite.config.js`:
```js
export default defineConfig({
  plugins: [vue()],
  server: {
    port: 8080,
    proxy: {
      '/api': {
        target: 'http://localhost:5000',
        changeOrigin: true,
      }
    }
  }
})
```

---

## Known Limitations

1. **No real-time WebSocket updates** — Stage progress is polled via REST API. Socket.IO event handlers are defined backend-side but not fully wired to the frontend.
2. **Code viewer is read-only** — No editing capability for intermediate files.
3. **File tree shows all workspace files** — Large file trees (100+ files) may be slow. Needs virtualization.
4. **No dark mode toggle** — Uses Element Plus default theme.
5. **No responsive mobile layout** — Designed for desktop use only.
