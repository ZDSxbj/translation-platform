<template>
  <div class="workspace-page">
    <!-- Header -->
    <div class="workspace-header">
      <el-button :icon="ArrowLeft" @click="$router.push('/')" text>Back to Home</el-button>
      <h2>Translation Pipeline</h2>
      <el-tag :type="statusTagType">{{ statusText }}</el-tag>
    </div>

    <!-- Stage Stepper -->
    <div class="stepper-section">
      <PipelineStepper
        :stages="store.stages"
        :current-index="store.currentStageIndex"
      />
    </div>

    <!-- Stage Panels — show all completed + current stage -->
    <div v-for="(stage, index) in visibleStages" :key="stage.id" class="stage-card">
      <div class="stage-card-header">
        <el-tag
          :type="stageStatusTag(stage.status)"
          size="small"
          effect="dark"
        >
          {{ stage.name.split(':')[0] }}
        </el-tag>
        <span class="stage-card-title">{{ stage.name.split(':').slice(1).join(':').trim() || stage.name }}</span>
        <StatusBadge :status="stage.status" style="margin-left: auto" />
      </div>

      <template v-if="stage.id === 'postprocess'">
        <!-- Post-process: only run button (pending) + final product (completed) — no logs, no intermediate files -->
        <div v-if="stage.status === 'pending' || stage.status === 'failed'" class="stage-actions">
          <el-button type="primary" :icon="VideoPlay" :loading="store.isRunning" @click="onRunStage(stage.id)">
            {{ stage.status === 'failed' ? 'Retry' : 'Run' }}
          </el-button>
        </div>
        <div v-if="stage.status === 'running'" class="stage-running">
          <el-alert type="info" :closable="false" show-icon title="Packaging output..." />
          <el-progress :percentage="100" :indeterminate="true" :stroke-width="6" style="margin-top:12px" />
        </div>
      </template>
      <StagePanel
        v-else
        :stage="stage"
        :session-id="store.sessionId"
        :is-running="stage.status === 'running'"
        :workspace-subdir="getWorkspaceSubdir(stage.id)"
        :hide-file-tree="true"
        @run="onRunStage"
      />

      <!-- Stage 1 Visualization -->
      <div v-if="stage.id === 'stage1_prep' && s1VizData.skeleton_files.length > 0" class="s1-viz-section">
        <div class="s1-viz-header">
          <h3>📊 Skeleton Analysis</h3>
          <el-button size="small" text @click="showS1Viz = !showS1Viz">{{ showS1Viz ? 'Hide' : 'Show' }}</el-button>
        </div>
        <template v-if="showS1Viz">
        <div class="s1-viz-tabs">
          <el-button size="small" :type="s1VizTab === 'skeleton' ? 'primary' : ''" @click="s1VizTab = 'skeleton'">Skeleton Files ({{ s1VizData.skeleton_files.length }})</el-button>
          <el-button size="small" :type="s1VizTab === 'callgraph' ? 'primary' : ''" @click="s1VizTab = 'callgraph'">Call Graph ({{ s1VizData.call_graph.nodes.length }} nodes)</el-button>
        </div>
        <!-- Skeleton files table -->
        <div v-if="s1VizTab === 'skeleton'" class="s1-skel-list">
          <div v-for="f in s1VizData.skeleton_files" :key="f.name" class="s1-skel-row" @click="viewSkelFile(f)" style="cursor:pointer">
            <el-icon class="comp-expand-icon" :class="{ rotated: s1ExpandedFile === f.name }"><ArrowRight /></el-icon>
            <span class="s1-skel-name">{{ f.name }}</span>
            <span class="s1-skel-stat">{{ f.fn_count }} fns</span>
            <span class="s1-skel-stat">{{ f.struct_count }} types</span>
            <el-tag v-if="f.opaque_count > 0" size="small" type="warning">{{ f.opaque_count }} opaque</el-tag>
            <span class="s1-skel-stat">{{ f.size }} lines</span>
          </div>
          <div v-if="s1SelectedFile" class="s1-skel-code">
            <div class="s1-skel-code-header">
              <span>📄 {{ s1SelectedFile.name }}</span>
              <el-button size="small" text @click="s1SelectedFile=null">✕</el-button>
            </div>
            <CodeViewer :content="s1SelectedFile.content" :language="s1SelectedFile.language" :filename="s1SelectedFile.name" />
          </div>
        </div>
        <!-- Call graph visualization -->
        <div v-if="s1VizTab === 'callgraph'" class="s1-callgraph">
          <p class="s1-cg-subtitle">Green nodes = internal | Orange nodes = external API calls</p>
          <CallGraph :nodes="s1VizData.call_graph.nodes" :edges="s1VizData.call_graph.edges" />
        </div>
        </template>
      </div>

      <!-- RAG Knowledge Cards — replace Stage 2 files -->
      <div v-if="stage.id === 'stage2_rag' && ragItems.length > 0" class="rag-knowledge-section">
        <div class="rag-header">
          <h3>🔍 RAG Knowledge Matches ({{ ragItems.length }} functions)</h3>
          <el-button size="small" text @click="showRagCards = !showRagCards">
            {{ showRagCards ? 'Hide' : 'Show' }}
          </el-button>
        </div>
        <template v-if="showRagCards">
        <p class="rag-subtitle">
          Jina Reranker matched C functions to known Rust translations in the knowledge base.
          These snippets are injected into the LLM prompt during Stage 3 translation.
        </p>
        <div v-if="ragLoading" class="rag-loading">
          <el-skeleton :rows="3" animated />
        </div>
        <div v-else class="rag-grid">
          <div v-for="item in ragItems" :key="item.func_file" class="rag-card">
            <div class="rag-card-header">
              <el-tag size="small" type="info">{{ item.func_file }}</el-tag>
            </div>
            <div class="rag-card-body">
              <div v-if="item.c_code" class="rag-snippet c-snippet">
                <div class="snippet-label">C Source</div>
                <pre><code>{{ item.c_code }}</code></pre>
              </div>
              <div v-if="item.rust_code" class="rag-snippet rust-snippet">
                <div class="snippet-label">Rust Match</div>
                <pre><code>{{ item.rust_code }}</code></pre>
              </div>
            </div>
          </div>
        </div>
        </template>
      </div>

      <!-- C/Rust Function Comparison — inside Stage 3 -->
      <div v-if="stage.id === 'stage3_translate' && comparisonData.functions.length > 0" class="comparison-section">
        <div class="comp-header">
          <h3>📝 Function Translation Results ({{ comparisonData.compiled }}/{{ comparisonData.functions.length }} compiled)</h3>
          <div class="comp-stats">
            <el-tag size="small" type="success">✓ {{ comparisonData.compiled - comparisonData.repaired }} one-shot</el-tag>
            <el-tag size="small" type="warning" style="margin-left:6px">🔧 {{ comparisonData.repaired }} repaired</el-tag>
            <el-tag size="small" type="danger" style="margin-left:6px">✗ {{ comparisonData.failed }} failed</el-tag>
            <el-button size="small" text style="margin-left:8px" @click="showComparison = !showComparison">
              {{ showComparison ? 'Hide' : 'Show' }}
            </el-button>
          </div>
        </div>
        <template v-if="showComparison">
        <div v-if="compLoading" class="comp-loading">
          <el-skeleton :rows="3" animated />
        </div>
        <div v-else class="comp-list">
          <div v-for="fn in comparisonData.functions" :key="fn.func_file" class="comp-card">
            <div class="comp-card-header" @click="fn._expanded = !fn._expanded" style="cursor:pointer">
              <el-icon class="comp-expand-icon" :class="{ rotated: fn._expanded }"><ArrowRight /></el-icon>
              <span class="comp-func-name">{{ fn.func_name || fn.func_file }}</span>
              <span :class="['comp-source-tag', 'tag-' + (fn.source_by || 'unknown')]">{{ fn.status_tag || 'unknown' }}</span>
              <span class="comp-func-file">{{ fn.rust_file }}</span>
            </div>
            <div v-if="fn._expanded" class="comp-card-body">
              <div class="comp-pane c-pane">
                <div class="comp-pane-label">C Source</div>
                <pre><code>{{ fn.c_code }}</code></pre>
              </div>
              <div class="comp-pane rust-pane">
                <div class="comp-pane-label">Rust Translation</div>
                <pre><code>{{ fn.rust_code || '// (unimplemented!)' }}</code></pre>
              </div>
            </div>
          </div>
        </div>
        </template>
      </div>
      <!-- Stage 4: Final Rust project file tree -->
      <div v-if="stage.id === 'postprocess' && stage.status === 'completed'" class="s4-output">
        <h4>📁 Translated Rust Project</h4>
        <div class="s4-tree">
          <el-tree
            :data="s4FileTree"
            :props="{ label: 'name', children: 'children' }"
            node-key="path"
            highlight-current
            @node-click="onS4FileClick"
            :expand-on-click-node="false"
            default-expand-all
            size="small"
          >
            <template #default="{ data }">
              <span class="tree-node" :class="{ 'is-folder': data.children !== undefined }">
                <el-icon v-if="data.children !== undefined" size="14"><Folder /></el-icon>
                <el-icon v-else size="14"><Document /></el-icon>
                <span>{{ data.name }}</span>
              </span>
            </template>
          </el-tree>
        </div>
        <div v-if="s4SelectedFile" class="s4-code">
          <CodeViewer :content="s4SelectedFile.content" :language="s4SelectedFile.language" :filename="s4SelectedFile.path" />
        </div>
      </div>
    </div>

    <!-- Navigation — shown below all stages -->
    <div class="nav-actions">
      <div v-if="store.allStagesComplete" class="all-done">
        <el-result icon="success" title="All Stages Complete" sub-title="Translation pipeline finished successfully.">
          <template #extra>
            <el-button type="success" :icon="Finished" size="large" @click="goToResults">
              View Results
            </el-button>
          </template>
        </el-result>
      </div>

      <div v-else-if="!store.isRunning && nextPendingStage" class="next-stage-hint">
        <p>Next: <strong>{{ nextPendingStage.name }}</strong></p>
        <el-button
          type="primary"
          :icon="VideoPlay"
          size="large"
          @click="onRunStage(nextPendingStage.id)"
        >
          Continue to {{ nextPendingStage.name.split(':')[0] }}
        </el-button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, watch } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { ArrowLeft, ArrowRight, VideoPlay, Finished, Folder, Document } from '@element-plus/icons-vue'
import { useTranslationStore } from '@/stores/translation'
import { getRagKnowledge, getFunctionComparison, getStage1Visualization, getOutputTree, getOutputFile, getWorkspaceFile } from '@/apis'
import CodeViewer from '@/components/repo/CodeViewer.vue'
import PipelineStepper from '@/components/pipeline/PipelineStepper.vue'
import CallGraph from '@/components/pipeline/CallGraph.vue'
import StagePanel from '@/components/pipeline/StagePanel.vue'
import StatusBadge from '@/components/common/StatusBadge.vue'

const props = defineProps({
  sessionId: { type: String, required: true },
})

const router = useRouter()
const store = useTranslationStore()

// RAG Knowledge display
const showRagCards = ref(false)
const ragLoading = ref(false)
const ragItems = ref([])

// Stage 4 final output
const s4FileTree = ref([])
const s4SelectedFile = ref(null)

async function loadS4Tree() {
  try {
    const res = await getOutputTree(store.sessionId, 'workspace/final_projects')
    s4FileTree.value = res.data?.data?.file_tree || []
  } catch (e) { s4FileTree.value = [] }
}

async function onS4FileClick(data) {
  if (data.children) return
  try {
    const res = await getOutputFile(store.sessionId, data.path)
    const content = res.data?.data?.content || ''
    s4SelectedFile.value = { content, language: data.language || 'rust', path: data.path }
  } catch (e) { s4SelectedFile.value = null }
}

watch(() => {
  const s4 = store.stages.find(s => s.id === 'postprocess')
  return s4?.status
}, async (ns) => { if (ns === 'completed') await loadS4Tree() })

// Stage 1 visualization
const showS1Viz = ref(false)
const s1VizData = ref({ skeleton_files: [], call_graph: { nodes: [], edges: [] }, preprocess_files: [], opaque_types: [] })
const s1VizLoading = ref(false)
const s1VizTab = ref('skeleton')
const s1SelectedFile = ref(null)
const s1ExpandedFile = ref(null)

async function viewSkelFile(f) {
  if (s1ExpandedFile.value === f.name) {
    s1ExpandedFile.value = null
    s1SelectedFile.value = null
    return
  }
  s1ExpandedFile.value = f.name
  try {
    const res = await getWorkspaceFile(store.sessionId, f.path)
    s1SelectedFile.value = {
      name: f.name,
      content: res.data?.data?.content || '',
      language: f.name.endsWith('.rs') ? 'rust' : 'plaintext',
    }
  } catch (e) {
    s1SelectedFile.value = { name: f.name, content: '// Failed to load', language: 'plaintext' }
  }
}

// Function comparison display (C/Rust side-by-side)
const showComparison = ref(false)
const comparisonData = ref({ functions: [], compiled: 0, repaired: 0, failed: 0 })
const compLoading = ref(false)

// Auto-fetch Stage 1 visualization
watch(() => {
  const s1 = store.stages.find(s => s.id === 'stage1_prep')
  return s1?.status
}, async (newStatus) => {
  if (newStatus === 'completed' && store.sessionId) {
    s1VizLoading.value = true
    showS1Viz.value = true
    try {
      const res = await getStage1Visualization(store.sessionId)
      s1VizData.value = res.data?.data || { skeleton_files: [], call_graph: { nodes: [], edges: [] }, preprocess_files: [], opaque_types: [] }
    } catch (e) {
      s1VizData.value = { skeleton_files: [], call_graph: { nodes: [], edges: [] }, preprocess_files: [], opaque_types: [] }
    } finally { s1VizLoading.value = false }
  }
}, { immediate: true })

// Auto-fetch RAG knowledge when Stage 2 completes
watch(() => {
  const s2 = store.stages.find(s => s.id === 'stage2_rag')
  return s2?.status
}, async (newStatus) => {
  if (newStatus === 'completed' && store.sessionId) {
    ragLoading.value = true
    showRagCards.value = true
    try {
      const res = await getRagKnowledge(store.sessionId)
      ragItems.value = res.data?.data?.knowledge || []
    } catch (e) {
      ragItems.value = []
    } finally {
      ragLoading.value = false
    }
  }
}, { immediate: true })

// Auto-fetch function comparison when Stage 3 completes OR on page load
watch(() => {
  const s3 = store.stages.find(s => s.id === 'stage3_translate')
  return s3?.status
}, async (newStatus) => {
  if (newStatus === 'completed' && store.sessionId) {
    await fetchComparison()
  }
}, { immediate: true })

async function fetchComparison() {
  compLoading.value = true
  showComparison.value = true
  try {
    const res = await getFunctionComparison(store.sessionId)
    comparisonData.value = res.data?.data || { functions: [], compiled: 0, repaired: 0, failed: 0 }
  } catch (e) {
    comparisonData.value = { functions: [], compiled: 0, repaired: 0, failed: 0 }
  } finally {
    compLoading.value = false
  }
}

onMounted(async () => {
  // On page refresh the store is empty — sync sessionId from the URL
  // first, then reload state from the backend.
  if (store.sessionId !== props.sessionId) {
    store.sessionId = props.sessionId
  }
  try {
    await store.refreshState()
  } catch (e) {
    // Session may have expired (backend restart, TTL, etc.)
  }
})

// Show all completed stages + the current active stage
const visibleStages = computed(() => {
  const stages = store.stages
  const idx = store.currentStageIndex
  const result = []
  // Show all completed stages + the currently running one
  for (let i = 0; i < stages.length; i++) {
    if (stages[i].status === 'completed' || stages[i].status === 'running' || i <= idx) {
      result.push(stages[i])
    }
  }
  return result
})

const nextPendingStage = computed(() => {
  const idx = store.currentStageIndex
  if (idx < store.stages.length) {
    const s = store.stages[idx]
    if (s.status === 'pending' || s.status === 'failed') return s
  }
  return null
})

const statusText = computed(() => {
  if (store.isRunning) return 'Running'
  if (store.allStagesComplete) return 'All Complete'
  if (store.hasFailedStage) return 'Stage Failed'
  return 'Ready'
})

const statusTagType = computed(() => {
  if (store.isRunning) return 'warning'
  if (store.allStagesComplete) return 'success'
  if (store.hasFailedStage) return 'danger'
  return 'info'
})

function stageStatusTag(status) {
  const map = { completed: 'success', running: 'warning', failed: 'danger', pending: 'info' }
  return map[status] || 'info'
}

function getWorkspaceSubdir(stageId) {
  const defaults = {
    stage1_prep: 'skeletons',
    stage2_rag: 'source_skeletons',
    stage3_translate: '',
    postprocess: 'final_projects',
    stage1_transpile: 'transpiled_raw',
    stage2_postprocess: 'transpiled',
  }
  return defaults[stageId] || ''
}

async function onRunStage(stageId) {
  if (!stageId) return
  try {
    const result = await store.runStageAction(stageId)
    ElMessage.success(`Stage completed: ${result?.summary || 'Done'}`)

    // Scroll to latest stage card
    setTimeout(() => {
      const cards = document.querySelectorAll('.stage-card')
      if (cards.length > 0) {
        cards[cards.length - 1].scrollIntoView({ behavior: 'smooth', block: 'center' })
      }
    }, 200)
  } catch (e) {
    ElMessage.error(e.response?.data?.message || e.message || 'Stage failed')
  }
}

function goToResults() {
  router.push(`/result/${store.sessionId}`)
}
</script>

<style scoped>
.workspace-page {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px 24px;
}
.workspace-header {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 20px;
}
.workspace-header h2 {
  flex: 1;
  margin: 0;
  font-size: 20px;
}
.stepper-section {
  background: #fff;
  border-radius: 12px;
  padding: 20px 24px;
  box-shadow: 0 2px 12px rgba(0,0,0,0.06);
  margin-bottom: 16px;
}
.stage-card {
  background: #fff;
  border-radius: 12px;
  padding: 16px 24px;
  box-shadow: 0 2px 12px rgba(0,0,0,0.06);
  margin-bottom: 16px;
}
.stage-card-header {
  display: flex;
  align-items: center;
  gap: 10px;
  padding-bottom: 12px;
  margin-bottom: 4px;
  border-bottom: 1px solid #ebeef5;
}
.stage-card-title {
  font-size: 15px;
  font-weight: 500;
  color: #606266;
}
.nav-actions {
  text-align: center;
  margin: 20px 0;
}
.next-stage-hint {
  background: #fff;
  border-radius: 12px;
  padding: 24px;
  box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}
.next-stage-hint p {
  margin: 0 0 16px;
  font-size: 15px;
  color: #606266;
}
.all-done {
  background: #fff;
  border-radius: 12px;
  padding: 16px;
  box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}

/* RAG Knowledge Cards */
.rag-knowledge-section {
  background: #fff;
  border-radius: 12px;
  padding: 20px 24px;
  margin-bottom: 16px;
  box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}
.rag-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 4px;
}
.rag-header h3 {
  margin: 0;
  font-size: 16px;
}
.rag-subtitle {
  color: #909399;
  font-size: 13px;
  margin: 4px 0 16px;
}
.rag-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
  gap: 12px;
}
.rag-card {
  border: 1px solid #ebeef5;
  border-radius: 8px;
  overflow: hidden;
  transition: box-shadow 0.2s;
}
.rag-card:hover {
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}
.rag-card-header {
  padding: 8px 12px;
  background: #f5f7fa;
  border-bottom: 1px solid #ebeef5;
}
.rag-card-body {
  padding: 8px 12px;
}
.rag-snippet {
  margin-bottom: 8px;
}
.rag-snippet:last-child {
  margin-bottom: 0;
}
.snippet-label {
  font-size: 11px;
  font-weight: 600;
  color: #909399;
  text-transform: uppercase;
  margin-bottom: 4px;
}
.rag-snippet pre {
  background: #f0f5ff;
  border-radius: 4px;
  padding: 8px 10px;
  margin: 0;
  font-size: 12px;
  line-height: 1.5;
  overflow-x: auto;
  max-height: 120px;
}
.rag-snippet pre code {
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  white-space: pre-wrap;
  word-break: break-all;
}
.rust-snippet pre {
  background: #f6ffed;
}
.rag-loading {
  padding: 16px 0;
}

/* Stage 1 Visualization */
.s1-viz-section {
  margin-top: 12px;
  padding: 16px 20px;
  background: #fafbfc;
  border-radius: 8px;
  border: 1px solid #ebeef5;
}
.s1-viz-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; }
.s1-viz-header h3 { margin: 0; font-size: 14px; }
.s1-viz-tabs { display: flex; gap: 8px; margin-bottom: 12px; }
.s1-skel-list { display: flex; flex-direction: column; gap: 6px; }
.s1-skel-row {
  display: flex; align-items: center; gap: 12px;
  padding: 8px 12px; background: #fff; border-radius: 6px;
  border: 1px solid #ebeef5; font-size: 13px;
}
.s1-skel-name { font-weight: 600; font-family: 'JetBrains Mono', monospace; min-width: 220px; color: #303133; }
.s1-skel-stat { color: #909399; font-size: 12px; }
.s1-cg-subtitle { color: #909399; font-size: 13px; margin-bottom: 8px; }

/* Function Comparison Cards */
.comparison-section {
  background: #fff;
  border-radius: 12px;
  padding: 20px 24px;
  margin-bottom: 16px;
  box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}
.comp-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 16px;
}
.comp-header h3 {
  margin: 0;
  font-size: 16px;
}
.comp-stats {
  display: flex;
  align-items: center;
}
.comp-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.comp-card {
  border: 1px solid #ebeef5;
  border-radius: 8px;
  overflow: hidden;
}
.comp-card-header {
  padding: 8px 12px;
  background: #f5f7fa;
  border-bottom: 1px solid #ebeef5;
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.comp-func-name {
  font-weight: 600;
  font-size: 14px;
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
}
.comp-func-file {
  font-size: 12px;
  color: #909399;
}
.comp-source-tag {
  font-size: 11px;
  font-weight: 600;
  padding: 2px 8px;
  border-radius: 10px;
  white-space: nowrap;
}
.tag-llm_one_shot { background: #e8f5e9; color: #2e7d32; }
.tag-llm_repaired { background: #fff3e0; color: #e65100; }
.tag-c2rust { background: #e3f2fd; color: #1565c0; }
.tag-failed { background: #ffebee; color: #c62828; }
.tag-unknown { background: #f5f5f5; color: #757575; }
.comp-card-body {
  display: grid;
  grid-template-columns: 1fr 1fr;
  min-height: 120px;
}
.comp-pane {
  padding: 10px 14px;
  overflow: hidden;
  min-width: 0;
}
.comp-pane:first-child {
  border-right: 1px solid #ebeef5;
}
.comp-pane-label {
  font-size: 11px;
  font-weight: 600;
  color: #909399;
  text-transform: uppercase;
  margin-bottom: 6px;
}
.c-pane pre {
  background: #f0f5ff;
}
.rust-pane pre {
  background: #f6ffed;
}
.comp-pane pre {
  border-radius: 4px;
  padding: 8px 10px;
  margin: 0;
  font-size: 12px;
  line-height: 1.5;
  overflow-y: auto;
  max-height: 500px;
}
.comp-pane pre code {
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  white-space: pre-wrap;
  word-break: break-all;
}
.comp-loading {
  padding: 16px 0;
}
.comp-expand-icon {
  transition: transform 0.2s;
  font-size: 14px;
  color: #909399;
}
.comp-expand-icon.rotated {
  transform: rotate(90deg);
}

/* Stage 4 output */
.s4-output { margin-top: 12px; padding: 12px 16px; background: #f0fdf4; border-radius: 8px; border: 1px solid #d9f99d; }
.s4-output h4 { margin: 0 0 8px; font-size: 14px; }
.s4-tree { background: #fff; border-radius: 6px; padding: 8px; max-height: 400px; overflow-y: auto; margin-bottom: 12px; }
.s4-code { background: #fff; border-radius: 6px; }

/* Skeleton file code viewer */
.s1-skel-code { margin-top: 10px; background: #fff; border: 1px solid #ebeef5; border-radius: 8px; overflow: hidden; }
.s1-skel-code-header { display: flex; align-items: center; justify-content: space-between; padding: 8px 14px; background: #f5f7fa; border-bottom: 1px solid #ebeef5; font-size: 13px; font-weight: 500; }
</style>
