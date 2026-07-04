<template>
  <div class="comparison-page" v-loading="loading">
    <!-- Header -->
    <div class="comparison-header">
      <el-button :icon="ArrowLeft" @click="goBack" text>Back to Workspace</el-button>
      <h2>Translation Comparison: His2Trans vs C2Rust</h2>
    </div>

    <!-- Metrics Dashboard -->
    <div class="metrics-section" v-if="metrics.length > 0">
      <h3 class="section-title">Quality Comparison</h3>
      <el-row :gutter="12">
        <el-col :xs="24" :sm="12" :md="8" :lg="6" v-for="m in metrics" :key="m.label" class="metric-col">
          <el-card shadow="hover" class="metric-card">
            <div class="metric-label">{{ m.label }}</div>
            <div class="metric-values">
              <div class="metric-row">
                <span class="engine-tag his2trans-tag">His2Trans</span>
                <span class="value" :class="{ winner: m.winner === 'his2trans' }">{{ m.his2trans }}</span>
              </div>
              <div class="metric-row">
                <span class="engine-tag c2rust-tag">C2Rust</span>
                <span class="value" :class="{ winner: m.winner === 'c2rust' }">{{ m.c2rust }}</span>
              </div>
            </div>
            <div class="metric-footer">
              <el-tag
                v-if="m.winner === 'his2trans'"
                type="success"
                size="small"
                effect="plain"
              >
                His2Trans Better
              </el-tag>
              <el-tag
                v-else-if="m.winner === 'c2rust'"
                type="warning"
                size="small"
                effect="plain"
              >
                C2Rust Better
              </el-tag>
              <span v-else class="neutral-label">—</span>
            </div>
          </el-card>
        </el-col>
      </el-row>
    </div>

    <!-- Side-by-Side File Browser -->
    <div class="files-section">
      <h3 class="section-title">Output Files</h3>
      <div class="dual-browser">
        <!-- His2Trans Pane -->
        <div class="browser-pane his2trans-pane">
          <div class="pane-header">
            <el-tag type="success" effect="dark" size="small">His2Trans</el-tag>
            <span class="pane-title">Output Files</span>
            <el-button size="small" @click="downloadHis2trans" :icon="Download">Download</el-button>
          </div>
          <div class="pane-body">
            <FileTree
              :tree="his2transTree"
              @file-selected="onHis2transFileSelected"
            />
            <CodeViewer
              v-if="activeHis2transFile"
              :content="activeHis2transFile.content"
              :language="activeHis2transFile.language || 'rust'"
              :filename="activeHis2transFile.path"
            />
            <div v-else class="no-file-hint">
              <el-empty description="Select a file to view" :image-size="60" />
            </div>
          </div>
        </div>

        <!-- C2Rust Pane -->
        <div class="browser-pane c2rust-pane">
          <div class="pane-header">
            <el-tag type="warning" effect="dark" size="small">C2Rust</el-tag>
            <span class="pane-title">Output Files</span>
            <el-button size="small" @click="downloadC2rust" :icon="Download">Download</el-button>
          </div>
          <div class="pane-body">
            <FileTree
              :tree="c2rustTree"
              @file-selected="onC2rustFileSelected"
            />
            <CodeViewer
              v-if="activeC2rustFile"
              :content="activeC2rustFile.content"
              :language="activeC2rustFile.language || 'rust'"
              :filename="activeC2rustFile.path"
            />
            <div v-else class="no-file-hint">
              <el-empty description="Select a file to view" :image-size="60" />
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { ArrowLeft, Download } from '@element-plus/icons-vue'
import FileTree from '@/components/repo/FileTree.vue'
import CodeViewer from '@/components/repo/CodeViewer.vue'
import { getReport, getOutputTree, getOutputFile, downloadResult } from '@/apis/index.js'

const props = defineProps({
  sessionIdA: { type: String, required: true },  // His2Trans session
  sessionIdB: { type: String, required: true },  // C2Rust session
})

const router = useRouter()
const loading = ref(true)

// Reports
const his2transReport = ref(null)
const c2rustReport = ref(null)

// File trees
const his2transTree = ref([])
const c2rustTree = ref([])

// Active files
const activeHis2transFile = ref(null)
const activeC2rustFile = ref(null)

// ------------------------------------------------------------------
// Metrics computation
// ------------------------------------------------------------------

const metrics = computed(() => {
  const h = his2transReport.value
  const c = c2rustReport.value

  if (!h || !c) return []

  // Extract His2Trans stats (from report details array)
  const hStats = extractHis2transStats(h)
  const cStats = extractC2rustStats(c)

  const items = []

  // Translation time
  const hTime = estimateTime(h)
  const cTime = estimateTime(c)
  items.push({
    label: 'Translation Time',
    his2trans: hTime || 'N/A',
    c2rust: cTime || 'N/A',
    winner: hTime && cTime ? (parseTime(hTime) < parseTime(cTime) ? 'his2trans' : 'c2rust') : null,
  })

  // Files generated
  const hFiles = hStats.rustFiles || 0
  const cFiles = cStats.total_files || 0
  items.push({
    label: 'Files Generated',
    his2trans: hFiles,
    c2rust: cFiles,
    winner: null,  // No clear "better"
  })

  // Compile success
  const hPassed = hStats.compilePassed || 0
  const hFailed = hStats.compileFailed || 0
  const hTotal = hPassed + hFailed || hFiles
  const hRate = hPassed + hFailed > 0 ? Math.round((hPassed / hTotal) * 100) + '%' : 'N/A'
  const cPassed = cStats.compile_passed || 0
  const cFailed = cStats.compile_failed || 0
  const cTotal = cPassed + cFailed
  const cRate = cTotal > 0 ? Math.round((cPassed / cTotal) * 100) + '%' : '0% (raw transpile)'
  items.push({
    label: 'Compile Success Rate',
    his2trans: hRate,
    c2rust: cRate,
    winner: hRate !== 'N/A' && cTotal > 0 && parseInt(hRate) > parseInt(cRate) ? 'his2trans' : null,
  })

  // Unsafe blocks + functions (C2Rust uses `unsafe fn` not `unsafe { }`)
  const hUnsafe = hStats.unsafeBlocks || 0
  const cUnsafe = (cStats.unsafe_blocks || 0) + (cStats.unsafe_functions || 0)
  items.push({
    label: 'Unsafe Items (blocks + fns)',
    his2trans: hUnsafe,
    c2rust: cUnsafe,
    winner: cUnsafe > 0 ? (hUnsafe < cUnsafe ? 'his2trans' : 'c2rust') : null,
  })

  // extern "C" functions
  const hExtern = hStats.externCFns || 0
  const cExtern = cStats.extern_c_functions || 0
  items.push({
    label: 'extern "C" Functions',
    his2trans: hExtern,
    c2rust: cExtern,
    winner: cExtern > 0 ? (hExtern < cExtern ? 'his2trans' : 'c2rust') : null,
  })

  // Raw pointer types
  const hPtrs = hStats.rawPtrs || 0
  const cPtrs = cStats.raw_ptr_types || 0
  items.push({
    label: 'Raw Pointer Types',
    his2trans: hPtrs,
    c2rust: cPtrs,
    winner: cPtrs > 0 ? (hPtrs < cPtrs ? 'his2trans' : 'c2rust') : null,
  })

  // Feature gates
  const hGates = hStats.featureGates || 0
  const cGates = cStats.feature_gates_removed || 0
  items.push({
    label: 'Feature Gates Needed',
    his2trans: hGates || '0 (stable Rust)',
    c2rust: cGates,
    winner: cGates > 0 ? 'his2trans' : null,
  })

  // Repair count
  items.push({
    label: 'LLM Repair Attempts',
    his2trans: hStats.repaired || 0,
    c2rust: 'N/A (mechanical)',
    winner: null,
  })

  return items
})

function extractHis2transStats(report) {
  const stats = {}
  // Try top-level stats first
  if (report.extracted_functions != null) stats.extracted = report.extracted_functions
  if (report.skeleton_rust_files != null) stats.skeletons = report.skeleton_rust_files
  if (report.translated_functions != null) stats.translated = report.translated_functions
  if (report.repaired != null) stats.repaired = report.repaired
  if (report.c2rust_fallback != null) stats.c2rustFallback = report.c2rust_fallback
  if (report.compile_passed != null) stats.compilePassed = report.compile_passed
  if (report.compile_failed != null) stats.compileFailed = report.compile_failed
  stats.compileTotal = (stats.compilePassed || 0) + (stats.compileFailed || 0)
  if (report.rag_signature_matches != null) stats.ragMatches = report.rag_signature_matches
  if (report.final_rust_files != null) stats.rustFiles = report.final_rust_files

  // Also scan stages for file counts, compile stats, and repair info
  if (report.stages) {
    for (const stage of report.stages) {
      if (stage.details) {
        if (stage.details.rust_files_count != null) stats.rustFiles = stage.details.rust_files_count
        if (stage.details.max_repair != null) stats.maxRepair = stage.details.max_repair
        // Compile stats may be in stage3_translate or postprocess
        if (stage.details.compile_passed != null) stats.compilePassed = (stats.compilePassed || 0) + stage.details.compile_passed
        if (stage.details.compile_failed != null) stats.compileFailed = (stats.compileFailed || 0) + stage.details.compile_failed
        if (stage.details.repaired != null) stats.repaired = (stats.repaired || 0) + stage.details.repaired
      }
    }
    // Postprocess stage often has aggregate repair count
    const pp = report.stages.find(s => s.id === 'postprocess')
    if (pp?.details) {
      if (pp.details.repaired != null && !stats.repaired) stats.repaired = pp.details.repaired
    }
  }

  // Fallback: count from output file tree or general numbers
  if (!stats.rustFiles && report.rust_files_generated) stats.rustFiles = report.rust_files_generated

  return stats
}

function extractC2rustStats(report) {
  // C2Rust report is structured differently — details from stage2
  if (report.stages) {
    const pp = report.stages.find(s => s.id === 'stage2_postprocess')
    if (pp?.details) return pp.details
  }
  // Fallback: try top-level keys
  return {
    total_files: report.total_files || 0,
    total_lines: report.total_lines || 0,
    unsafe_blocks: report.unsafe_blocks || 0,
    unsafe_functions: report.unsafe_functions || 0,
    extern_c_functions: report.extern_c_functions || 0,
    raw_ptr_types: report.raw_ptr_types || 0,
    feature_gates_removed: report.feature_gates_removed || 0,
  }
}

function estimateTime(report) {
  if (report.stages) {
    const timestamps = report.stages
      .filter(s => s.end_time)
      .map(s => new Date(s.end_time).getTime())
    const startTimes = report.stages
      .filter(s => s.start_time)
      .map(s => new Date(s.start_time).getTime())
    if (startTimes.length > 0 && timestamps.length > 0) {
      const start = Math.min(...startTimes)
      const end = Math.max(...timestamps)
      const diffSec = Math.round((end - start) / 1000)
      if (diffSec < 60) return `${diffSec}s`
      if (diffSec < 3600) return `${Math.round(diffSec / 60)}m ${diffSec % 60}s`
      return `${Math.floor(diffSec / 3600)}h ${Math.round((diffSec % 3600) / 60)}m`
    }
  }
  return null
}

function parseTime(str) {
  // Parse "3m 39s", "3s", "2h 5m" etc.
  const hMatch = str.match(/(\d+)\s*h/)
  const mMatch = str.match(/(\d+)\s*m/)
  const sMatch = str.match(/(\d+)\s*s/)
  let total = 0
  if (hMatch) total += parseInt(hMatch[1]) * 3600
  if (mMatch) total += parseInt(mMatch[1]) * 60
  if (sMatch) total += parseInt(sMatch[1])
  return total
}

// ------------------------------------------------------------------
// File loading
// ------------------------------------------------------------------

async function onHis2transFileSelected(node) {
  try {
    const res = await getOutputFile(props.sessionIdA, node.path)
    activeHis2transFile.value = res.data?.data
  } catch {
    ElMessage.error('Failed to load file')
  }
}

async function onC2rustFileSelected(node) {
  try {
    const res = await getOutputFile(props.sessionIdB, node.path)
    activeC2rustFile.value = res.data?.data
  } catch {
    ElMessage.error('Failed to load file')
  }
}

async function downloadHis2trans() {
  try {
    const res = await downloadResult(props.sessionIdA)
    const url = URL.createObjectURL(res.data)
    const a = document.createElement('a')
    a.href = url
    a.download = `translated_his2trans.zip`
    a.click()
    URL.revokeObjectURL(url)
  } catch {
    ElMessage.error('Download failed')
  }
}

async function downloadC2rust() {
  try {
    const res = await downloadResult(props.sessionIdB)
    const url = URL.createObjectURL(res.data)
    const a = document.createElement('a')
    a.href = url
    a.download = `translated_c2rust.zip`
    a.click()
    URL.revokeObjectURL(url)
  } catch {
    ElMessage.error('Download failed')
  }
}

function goBack() {
  router.push(`/workspace/${props.sessionIdA}?compare=true&c2rust_session=${props.sessionIdB}`)
}

// ------------------------------------------------------------------
// Init
// ------------------------------------------------------------------

onMounted(async () => {
  try {
    const [hReport, cReport, hTree, cTree] = await Promise.all([
      getReport(props.sessionIdA).then(r => r.data?.data).catch(() => null),
      getReport(props.sessionIdB).then(r => r.data?.data).catch(() => null),
      getOutputTree(props.sessionIdA).then(r => r.data?.data?.file_tree || []).catch(() => []),
      getOutputTree(props.sessionIdB, 'transpiled').then(r => r.data?.data?.file_tree || []).catch(() => []),
    ])

    his2transReport.value = hReport
    c2rustReport.value = cReport
    his2transTree.value = hTree
    c2rustTree.value = cTree
  } catch (e) {
    ElMessage.error('Failed to load comparison data')
  } finally {
    loading.value = false
  }
})
</script>

<style scoped>
.comparison-page {
  max-width: 1400px;
  margin: 0 auto;
  padding: 20px 24px;
}

.comparison-header {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 24px;
}

.comparison-header h2 {
  flex: 1;
  margin: 0;
  font-size: 22px;
  color: #303133;
}

.section-title {
  font-size: 17px;
  font-weight: 600;
  color: #303133;
  margin: 0 0 14px;
}

.metrics-section {
  margin-bottom: 24px;
}

.metric-col {
  margin-bottom: 12px;
}

.metric-card {
  border-radius: 10px;
}

.metric-label {
  font-size: 13px;
  font-weight: 600;
  color: #606266;
  margin-bottom: 10px;
  min-height: 36px;
}

.metric-values {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin-bottom: 8px;
}

.metric-row {
  display: flex;
  align-items: center;
  gap: 8px;
}

.engine-tag {
  font-size: 11px;
  font-weight: 600;
  padding: 1px 6px;
  border-radius: 3px;
  min-width: 64px;
  text-align: center;
}
.his2trans-tag { background: #e1f3d8; color: #529b2e; }
.c2rust-tag { background: #faecd8; color: #e6a23c; }

.value {
  font-family: 'JetBrains Mono', monospace;
  font-size: 14px;
  color: #303133;
}
.value.winner {
  font-weight: 700;
  color: #529b2e;
}

.metric-footer {
  min-height: 24px;
}
.neutral-label {
  font-size: 12px;
  color: #c0c4cc;
}

.files-section {
  margin-top: 8px;
}

.dual-browser {
  display: flex;
  gap: 12px;
  min-height: 500px;
}

.browser-pane {
  flex: 1;
  background: #fff;
  border-radius: 12px;
  box-shadow: 0 2px 12px rgba(0,0,0,0.06);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.pane-header {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 14px;
  border-bottom: 1px solid #ebeef5;
  background: #fafafa;
}
.pane-title {
  flex: 1;
  font-size: 14px;
  font-weight: 600;
  color: #303133;
}

.pane-body {
  display: flex;
  flex: 1;
  overflow: hidden;
}

.pane-body > :first-child {
  width: 240px;
  flex-shrink: 0;
  border-right: 1px solid #ebeef5;
  overflow-y: auto;
}

.pane-body > :last-child {
  flex: 1;
  overflow: hidden;
}

.no-file-hint {
  display: flex;
  align-items: center;
  justify-content: center;
  flex: 1;
}

@media (max-width: 900px) {
  .dual-browser {
    flex-direction: column;
  }
}
</style>
