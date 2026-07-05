<template>
  <div class="home-page">
    <div class="hero-section">
      <h1 class="hero-title">Code Translation Platform</h1>
      <p class="hero-subtitle">C/C++ → Rust Repository Translation with Full Pipeline Visualization</p>
    </div>

    <!-- Step 1: Upload -->
    <div class="card upload-card">
      <div class="card-header">
        <el-icon><UploadFilled /></el-icon>
        <span>Step 1: Upload Repository</span>
        <el-tag v-if="uploadResult" type="success" size="small" effect="plain">done</el-tag>
      </div>
      <ZipUploader
        @uploaded="onUploaded"
        @error="onUploadError"
        :disabled="translating"
      />
      <div v-if="uploadResult" class="upload-result">
        <el-alert
          :title="`Uploaded: ${uploadResult.stats?.file_count || 0} files, ${uploadResult.stats?.dir_count || 0} directories`"
          type="success"
          :closable="false"
          show-icon
        />
        <div v-if="uploadResult.stats?.languages" class="lang-tags">
          <el-tag v-for="(count, lang) in uploadResult.stats.languages" :key="lang"
            size="small" type="info">{{ lang }}: {{ count }}</el-tag>
        </div>
      </div>
    </div>

    <!-- Step 2: Project Analysis (appears after upload) -->
    <div v-if="uploadResult" class="card analysis-card">
      <div class="card-header">
        <el-icon><DataAnalysis /></el-icon>
        <span>Step 2: Project Analysis</span>
        <el-tag v-if="analysisResult" :type="analysisResult?.project_type === 'standard_c' ? 'success' : 'warning'" size="small" effect="plain">
          {{ analysisResult?.project_type === 'standard_c' ? 'Standard C' : analysisResult?.project_type === 'ohos' ? 'OHOS' : 'analyzing...' }}
        </el-tag>
      </div>
      <AnalysisPanel
        :project-id="uploadResult.project_id"
        v-model="analysisConfig"
        @analysis-complete="onAnalysisComplete"
      />
    </div>

    <!-- Step 3: Configure & Start (appears after analysis) -->
    <div v-if="analysisReady" class="card config-card">
      <div class="card-header">
        <el-icon><Setting /></el-icon>
        <span>Step 3: Configure & Start Translation</span>
        <el-tag type="info" size="small" effect="plain">{{ config.engine || 'his2trans' }}</el-tag>
      </div>

      <div class="config-layout">
        <div class="config-main">
          <ConfigPanel
            v-model="config"
            :disabled="translating"
            :project-type="analysisResult?.project_type || null"
          />
          <div class="start-section">
            <el-button
              type="primary"
              size="large"
              :icon="VideoPlay"
              :disabled="!canStart || translating"
              :loading="translating"
              @click="onStartTranslation"
            >
              {{ translating ? 'Starting...' : 'Start Translation' }}
            </el-button>
          </div>
        </div>
      </div>
    </div>

    <!-- Source Repo Preview (always shown after upload) -->
    <div v-if="uploadResult" class="preview-section">
      <div class="preview-header">
        <el-icon><FolderOpened /></el-icon>
        <span>Source Repository Preview</span>
      </div>
      <div class="preview-pane">
        <FileTree
          :tree="uploadResult.file_tree || []"
          @file-selected="onSourceFileSelected"
        />
        <CodeViewer
          v-if="selectedSourceFile"
          :content="selectedSourceFile.content"
          :language="selectedSourceFile.language"
          :filename="selectedSourceFile.path"
        />
        <div v-else class="no-file-selected">
          <el-empty description="Select a file to preview" :image-size="80" />
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import {
  UploadFilled, Setting, VideoPlay, FolderOpened, DataAnalysis,
} from '@element-plus/icons-vue'
import ZipUploader from '@/components/upload/ZipUploader.vue'
import ConfigPanel from '@/components/config/ConfigPanel.vue'
import AnalysisPanel from '@/components/config/AnalysisPanel.vue'
import FileTree from '@/components/repo/FileTree.vue'
import CodeViewer from '@/components/repo/CodeViewer.vue'
import { uploadZip, getProjectFile } from '@/apis/index.js'
import { useTranslationStore } from '@/stores/translation'

const router = useRouter()
const store = useTranslationStore()

const translating = ref(false)
const uploadResult = ref(null)
const analysisResult = ref(null)
const selectedSourceFile = ref(null)
const config = ref({
  engine: 'his2trans',
  model: 'deepseek-chat',
  use_rag: false,
  max_repair: 3,
  ohos_root: '',
  extra_includes: [],
})
const analysisConfig = ref({
  ohos_root: '',
})

const analysisReady = computed(() => analysisResult.value !== null)
const canStart = computed(() => uploadResult.value !== null && analysisReady.value)

function onUploaded(result) {
  uploadResult.value = result
  ElMessage.success('Repository uploaded successfully')
}

function onUploadError(msg) {
  ElMessage.error(msg)
}

function onAnalysisComplete(result) {
  analysisResult.value = result
  // If project type is standard_c, auto-fill config
  if (result?.project_type === 'standard_c') {
    config.value.ohos_root = ''
  }
}

async function onSourceFileSelected(node) {
  if (!uploadResult.value) return
  try {
    const res = await getProjectFile(uploadResult.value.project_id, node.path)
    selectedSourceFile.value = res.data?.data
  } catch (e) {
    ElMessage.error('Failed to load file')
  }
}

async function onStartTranslation() {
  if (!canStart.value) return
  translating.value = true

  try {
    // Merge analysis config into translation config
    const fullConfig = {
      ...config.value,
      ohos_root: analysisConfig.value.ohos_root || config.value.ohos_root,
    }
    await store.initSession(uploadResult.value.project_id, fullConfig)
    router.push(`/workspace/${store.sessionId}`)
  } catch (e) {
    ElMessage.error(e.response?.data?.message || 'Failed to start translation')
    translating.value = false
  }
}
</script>

<style scoped>
.home-page {
  max-width: 1000px;
  margin: 0 auto;
  padding: 24px;
}
.hero-section {
  text-align: center;
  padding: 32px 0 24px;
}
.hero-title {
  font-size: 28px;
  font-weight: 700;
  color: #303133;
  margin: 0 0 8px;
}
.hero-subtitle {
  font-size: 14px;
  color: #909399;
  margin: 0;
}
.card {
  background: #fff;
  border-radius: 12px;
  padding: 20px;
  box-shadow: 0 2px 12px rgba(0,0,0,0.06);
  margin-bottom: 20px;
}
.card-header {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 16px;
  font-weight: 600;
  color: #303133;
  margin-bottom: 16px;
  padding-bottom: 12px;
  border-bottom: 1px solid #ebeef5;
}
.card-header .el-tag {
  margin-left: auto;
}
.upload-result {
  margin-top: 12px;
}
.lang-tags {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
  margin-top: 8px;
}
.config-layout {
  display: flex;
  gap: 0;
}
.config-main {
  flex: 1;
}
.start-section {
  margin-top: 20px;
  text-align: center;
}
.preview-section {
  background: #fff;
  border-radius: 12px;
  box-shadow: 0 2px 12px rgba(0,0,0,0.06);
  overflow: hidden;
}
.preview-header {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 14px 20px;
  font-size: 15px;
  font-weight: 600;
  background: #fafafa;
  border-bottom: 1px solid #ebeef5;
}
.preview-pane {
  display: flex;
  height: 400px;
}
.preview-pane > :first-child {
  width: 280px;
  flex-shrink: 0;
  border-right: 1px solid #ebeef5;
  overflow-y: auto;
}
.preview-pane > :last-child {
  flex: 1;
  overflow: hidden;
}
.no-file-selected {
  display: flex;
  align-items: center;
  justify-content: center;
  flex: 1;
}
</style>
