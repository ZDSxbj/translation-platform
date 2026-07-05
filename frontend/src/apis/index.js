import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 600000,  // 10 minutes — stages can take several minutes
})

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    const msg = error.response?.data?.message || error.message || 'Network error'
    console.error('[API]', msg)
    return Promise.reject(error)
  }
)

// ---- Upload ----
export function uploadZip(file) {
  const form = new FormData()
  form.append('file', file)
  return api.post('/upload/zip', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 120000,
  })
}

// ---- Project ----
export function getProjectTree(projectId) {
  return api.get(`/project/${projectId}/tree`)
}

export function getProjectFile(projectId, path) {
  return api.get(`/project/${projectId}/file`, { params: { path } })
}

export function getProjectStats(projectId) {
  return api.get(`/project/${projectId}/stats`)
}

export function getCompileCommands(projectId) {
  return api.get(`/project/${projectId}/compile_commands`)
}

export function analyzeProject(projectId) {
  return api.get(`/project/${projectId}/analyze`)
}

export function fixProjectPaths(projectId) {
  return api.post(`/project/${projectId}/fix-paths`)
}

// ---- Translate ----
export function startTranslation(projectId, config) {
  return api.post('/translate/start', { project_id: projectId, ...config })
}

export function getSessionState(sessionId) {
  return api.get(`/translate/${sessionId}/state`)
}

export function runStage(sessionId, stageId) {
  return api.post(`/translate/${sessionId}/stage/${stageId}/run`, {}, {
    timeout: 1800000,  // 30 minutes — Stage 3 translation can take 10+ minutes
  })
}

export function getStageResult(sessionId, stageId) {
  return api.get(`/translate/${sessionId}/stage/${stageId}/result`)
}

export function getStageLogs(sessionId, stageId) {
  return api.get(`/translate/${sessionId}/stage/${stageId}/logs`)
}

export function getOutputTree(sessionId, subdir = '') {
  return api.get(`/translate/${sessionId}/output/tree`, { params: subdir ? { subdir } : {} })
}

export function getOutputFile(sessionId, path) {
  return api.get(`/translate/${sessionId}/output/file`, { params: { path } })
}

// ---- Workspace (intermediate results) ----
export function getWorkspaceTree(sessionId, subdir = '') {
  return api.get(`/translate/${sessionId}/workspace/tree`, { params: { subdir } })
}

export function getWorkspaceFile(sessionId, path) {
  return api.get(`/translate/${sessionId}/workspace/file`, { params: { path } })
}

// ---- Download ----
export function downloadResult(sessionId) {
  return api.get(`/download/${sessionId}/result`, { responseType: 'blob' })
}

export function getReport(sessionId) {
  return api.get(`/translate/${sessionId}/report`)
}

export function downloadReport(sessionId) {
  return api.get(`/download/${sessionId}/report`, { responseType: 'blob' })
}

// ---- RAG Knowledge ----
export function getRagKnowledge(sessionId) {
  return api.get(`/translate/${sessionId}/rag/knowledge`)
}

// ---- Function Comparison ----
export function getFunctionComparison(sessionId) {
  return api.get(`/translate/${sessionId}/functions/comparison`)
}

// ---- Stage 1 Visualization ----
export function getStage1Visualization(sessionId) {
  return api.get(`/translate/${sessionId}/stage/stage1_prep/visualization`)
}
