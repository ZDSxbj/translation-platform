<template>
  <div class="result-page">
    <!-- Header -->
    <div class="result-header">
      <div class="header-left">
        <el-button :icon="ArrowLeft" @click="goToWorkspace" text>Back to Pipeline</el-button>
        <h2>Translation Results</h2>
        <el-tag type="success">Complete</el-tag>
      </div>
      <div class="header-right">
        <el-button :icon="Download" type="primary" @click="onDownloadResult">
          Download Result ZIP
        </el-button>
        <el-button :icon="Document" @click="onDownloadReport">
          Download Report JSON
        </el-button>
      </div>
    </div>

    <!-- Pipeline Summary -->
    <div v-if="report" class="summary-section">
      <h3>Pipeline Summary</h3>
      <el-descriptions :column="3" border size="small">
        <el-descriptions-item label="Engine">{{ report.config?.engine || 'his2trans' }}</el-descriptions-item>
        <el-descriptions-item v-if="!isC2Rust" label="Model">{{ report.config?.model || '-' }}</el-descriptions-item>
        <el-descriptions-item v-if="!isC2Rust" label="RAG">{{ report.config?.use_rag ? 'Enabled' : 'Disabled' }}</el-descriptions-item>
        <el-descriptions-item v-if="!isC2Rust" label="Max Repair Rounds">{{ report.config?.max_repair || '-' }}</el-descriptions-item>
        <el-descriptions-item label="OHOS Root">{{ report.config?.ohos_root || '(none)' }}</el-descriptions-item>
        <el-descriptions-item label="Generated At">{{ formatTime(report.generated_at) }}</el-descriptions-item>
      </el-descriptions>
    </div>

    <!-- Stage-by-Stage Results -->
    <div v-if="report?.stages?.length" class="stages-section">
      <h3>Stage Details</h3>
      <div v-for="stage in report.stages" :key="stage.id" class="stage-detail-card">
        <div class="stage-card-head">
          <el-tag :type="stage.status === 'completed' ? 'success' : stage.status === 'failed' ? 'danger' : 'info'" size="small">
            {{ stage.name.split(':')[0] }}
          </el-tag>
          <span class="stage-card-name">{{ stage.name.split(':').slice(1).join(':').trim() }}</span>
          <StatusBadge :status="stage.status" style="margin-left: auto" />
        </div>
        <div v-if="stage.summary" class="stage-summary">{{ stage.summary }}</div>
        <div v-if="stage.details" class="stage-details">
          <el-descriptions :column="4" size="small" border>
            <template v-for="(val, key) in stage.details" :key="key">
              <el-descriptions-item :label="formatKey(key)">{{ val }}</el-descriptions-item>
            </template>
          </el-descriptions>
        </div>
        <div class="stage-timing" v-if="stage.start_time">
          <span>Started: {{ formatTime(stage.start_time) }}</span>
          <span v-if="stage.end_time"> — Ended: {{ formatTime(stage.end_time) }}</span>
          <span v-if="stage.log_count"> | Logs: {{ stage.log_count }} entries</span>
        </div>
      </div>
    </div>

    <!-- Pipeline Statistics -->
    <div v-if="hasStats" class="stats-section">
      <h3>Pipeline Statistics</h3>
      <!-- His2Trans stats -->
      <el-row v-if="!isC2Rust" :gutter="16">
        <el-col :span="6" v-if="report.extracted_functions != null">
          <el-statistic title="Extracted Functions" :value="report.extracted_functions" />
        </el-col>
        <el-col :span="6" v-if="report.skeleton_rust_files != null">
          <el-statistic title="Skeleton Rust Files" :value="report.skeleton_rust_files" />
        </el-col>
        <el-col :span="6" v-if="report.signature_matches != null">
          <el-statistic title="RAG Signature Matches" :value="report.signature_matches" />
        </el-col>
        <el-col :span="6" v-if="report.translated_functions != null">
          <el-statistic title="Translated Functions" :value="report.translated_functions" />
        </el-col>
        <el-col :span="6" v-if="report.compile_passed != null">
          <el-statistic title="Compile Passed" :value="report.compile_passed">
            <template #suffix>
              <span v-if="report.compile_failed" class="stat-suffix-fail">/ {{ report.compile_passed + report.compile_failed }} total</span>
            </template>
          </el-statistic>
        </el-col>
        <el-col :span="6" v-if="report.final_rust_files != null">
          <el-statistic title="Final Rust Files" :value="report.final_rust_files" />
        </el-col>
        <el-col :span="6" v-if="report.total_effective_lines != null">
          <el-statistic title="Effective Lines" :value="report.total_effective_lines">
            <template #suffix>
              <span class="stat-suffix-muted">(excl. comments)</span>
            </template>
          </el-statistic>
        </el-col>
        <el-col :span="6" v-if="report.unsafe_blocks != null">
          <el-statistic title="Unsafe Blocks" :value="report.unsafe_blocks">
            <template #suffix>
              <span class="stat-suffix-unsafe">{{ report.unsafe_effective_lines }} lines</span>
            </template>
          </el-statistic>
        </el-col>
        <el-col :span="6" v-if="report.unsafe_ratio != null">
          <el-statistic title="Unsafe Ratio" :value="report.unsafe_ratio + '%'">
            <template #suffix>
              <span class="stat-suffix-muted">{{ report.unsafe_effective_lines }} / {{ report.total_effective_lines }}</span>
            </template>
          </el-statistic>
        </el-col>
      </el-row>
      <!-- C2Rust stats -->
      <el-row v-else :gutter="16">
        <el-col :span="6" v-if="c2rustStats.total_files != null">
          <el-statistic title="Files Generated" :value="c2rustStats.total_files" />
        </el-col>
        <el-col :span="6" v-if="c2rustStats.total_lines != null">
          <el-statistic title="Total Lines" :value="c2rustStats.total_lines" />
        </el-col>
        <el-col :span="6" v-if="c2rustStats.unsafe_blocks != null">
          <el-statistic title="Unsafe Items" :value="c2rustStats.unsafe_blocks" />
        </el-col>
        <el-col :span="6" v-if="c2rustStats.unsafe_body_lines != null">
          <el-statistic title="Unsafe Lines" :value="c2rustStats.unsafe_body_lines">
            <template #suffix>
              <span style="font-size:12px;color:#909399">/ {{ c2rustStats.total_lines }} total</span>
            </template>
          </el-statistic>
        </el-col>
        <el-col :span="6" v-if="c2rustStats.extern_c_functions != null">
          <el-statistic title="extern C Functions" :value="c2rustStats.extern_c_functions" />
        </el-col>
        <el-col :span="6" v-if="c2rustStats.raw_ptr_types != null">
          <el-statistic title="Raw Pointer Types" :value="c2rustStats.raw_ptr_types" />
        </el-col>
        <el-col :span="6" v-if="c2rustStats.compile_passed != null">
          <el-statistic title="Compile Passed" :value="c2rustStats.compile_passed">
            <template #suffix>
              <span v-if="c2rustStats.compile_failed" class="stat-suffix-fail">/ {{ c2rustStats.compile_passed + c2rustStats.compile_failed }} total</span>
            </template>
          </el-statistic>
        </el-col>
      </el-row>
    </div>

    <!-- Main content: Tree + Code Viewer -->
    <div class="result-body-section">
      <h3>Translated Repository</h3>
      <div class="result-body">
        <div class="tree-panel">
          <div class="panel-title">
            <el-icon><FolderOpened /></el-icon>
            <span>Files</span>
          </div>
          <FileTree
            :tree="fileTree"
            @file-selected="onFileSelected"
          />
        </div>
        <div class="code-panel">
          <CodeViewer
            v-if="selectedFile"
            :content="selectedFile.content"
            :language="selectedFile.language"
            :filename="selectedFile.path"
          />
          <div v-else class="no-file">
            <el-empty description="Select a file to view" :image-size="100" />
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
import { ArrowLeft, Download, Document, FolderOpened } from '@element-plus/icons-vue'
import { useTranslationStore } from '@/stores/translation'
import { downloadResult, downloadReport, getReport } from '@/apis/index.js'
import FileTree from '@/components/repo/FileTree.vue'
import CodeViewer from '@/components/repo/CodeViewer.vue'
import StatusBadge from '@/components/common/StatusBadge.vue'

const props = defineProps({
  sessionId: { type: String, required: true },
})

const router = useRouter()
const store = useTranslationStore()

const fileTree = ref([])
const selectedFile = ref(null)
const report = ref(null)

const isC2Rust = computed(() => report.value?.config?.engine === 'c2rust')

const c2rustStats = computed(() => {
  if (!isC2Rust.value || !report.value?.stages) return {}
  const pp = report.value.stages.find(s => s.id === 'stage2_postprocess')
  return pp?.details || {}
})

const hasStats = computed(() => {
  if (!report.value) return false
  if (isC2Rust.value) {
    // C2Rust stats are in stage2_postprocess details
    const pp = report.value.stages?.find(s => s.id === 'stage2_postprocess')
    return pp?.details != null && Object.keys(pp.details).length > 0
  }
  const keys = ['extracted_functions', 'skeleton_rust_files', 'signature_matches',
    'translated_functions', 'compile_passed', 'final_rust_files']
  return keys.some(k => report.value[k] != null)
})

onMounted(async () => {
  // Sync store to the current session (important for C2Rust vs His2Trans)
  store.sessionId = props.sessionId
  try {
    await store.refreshState()
  } catch (e) {
    // Session state might not be available (backend restarted), but we
    // can still load the report and output tree.
  }

  // Load report from backend
  try {
    const res = await getReport(props.sessionId)
    report.value = res.data?.data || null
  } catch (e) {
    console.error('Failed to load report:', e)
  }

  // Load output tree — C2Rust stores output in transpiled/ not workspace/final_projects
  try {
    const subdir = isC2Rust.value ? 'transpiled' : ''
    const tree = await store.getOutputTree(subdir)
    fileTree.value = tree
  } catch (e) {
    console.error('Failed to load output tree:', e)
  }
})

function formatTime(iso) {
  if (!iso) return '-'
  try {
    const d = new Date(iso)
    return d.toLocaleString()
  } catch { return iso }
}

function formatKey(key) {
  return key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
}

async function onFileSelected(node) {
  try {
    // Tree API paths already include the subdir prefix (e.g.
    // "transpiled/src/foo.rs"), so no extra prefix is needed.
    const data = await store.getOutputFileContent(node.path)
    selectedFile.value = data
  } catch (e) {
    ElMessage.error('Failed to load file')
  }
}

function goToWorkspace() {
  router.push(`/workspace/${store.sessionId}`)
}

async function onDownloadResult() {
  try {
    const res = await downloadResult(store.sessionId)
    const blob = new Blob([res.data], { type: 'application/zip' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `translated_${store.sessionId.slice(0, 8)}.zip`
    a.click()
    URL.revokeObjectURL(url)
    ElMessage.success('Download started')
  } catch (e) {
    ElMessage.error('Download failed')
  }
}

async function onDownloadReport() {
  try {
    const res = await downloadReport(store.sessionId)
    const blob = new Blob([res.data], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `report_${store.sessionId.slice(0, 8)}.json`
    a.click()
    URL.revokeObjectURL(url)
    ElMessage.success('Report downloaded')
  } catch (e) {
    ElMessage.error('Download failed')
  }
}
</script>

<style scoped>
.result-page {
  max-width: 1400px;
  margin: 0 auto;
  padding: 20px 24px;
}
.result-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 20px;
  flex-wrap: wrap;
  gap: 12px;
}
.header-left {
  display: flex;
  align-items: center;
  gap: 12px;
}
.header-left h2 {
  margin: 0;
  font-size: 20px;
}
.header-right {
  display: flex;
  gap: 8px;
}

.summary-section, .stages-section, .stats-section, .result-body-section {
  background: #fff;
  border-radius: 12px;
  padding: 20px 24px;
  box-shadow: 0 2px 12px rgba(0,0,0,0.06);
  margin-bottom: 16px;
}
.summary-section h3, .stages-section h3, .stats-section h3, .result-body-section h3 {
  margin: 0 0 16px;
  font-size: 16px;
}

.stage-detail-card {
  border: 1px solid #ebeef5;
  border-radius: 8px;
  padding: 12px 16px;
  margin-bottom: 12px;
}
.stage-card-head {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 8px;
}
.stage-card-name {
  font-size: 14px;
  font-weight: 500;
  color: #303133;
}
.stage-summary {
  font-size: 13px;
  color: #606266;
  margin-bottom: 8px;
}
.stage-details {
  margin-bottom: 8px;
}
.stage-timing {
  font-size: 12px;
  color: #909399;
}
.stat-suffix-fail {
  font-size: 12px;
  color: #909399;
}
.stat-suffix-unsafe {
  font-size: 12px;
  color: #e6a23c;
}
.stat-suffix-muted {
  font-size: 12px;
  color: #909399;
}

.result-body {
  display: flex;
  border: 1px solid #ebeef5;
  border-radius: 8px;
  overflow: hidden;
  height: 550px;
}
.tree-panel {
  width: 300px;
  flex-shrink: 0;
  border-right: 1px solid #ebeef5;
  display: flex;
  flex-direction: column;
  overflow-y: auto;
}
.panel-title {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px 16px;
  font-weight: 600;
  border-bottom: 1px solid #ebeef5;
  background: #fafafa;
}
.code-panel {
  flex: 1;
  overflow: hidden;
}
.no-file {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
}

.el-row {
  margin: 0 !important;
}
</style>
