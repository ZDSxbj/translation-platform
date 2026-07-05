<template>
  <div class="stage-panel">
    <div class="stage-info">
      <div class="stage-title-row">
        <StatusBadge :status="stage.status" />
        <h4>{{ stage.name }}</h4>
      </div>
      <p v-if="stage.description" class="stage-desc">{{ stage.description }}</p>
    </div>

    <!-- Action buttons -->
    <div class="stage-actions" v-if="stage.status === 'pending' || stage.status === 'failed'">
      <el-button
        type="primary"
        :icon="VideoPlay"
        :loading="isRunning"
        @click="$emit('run', stage.id)"
      >
        {{ stage.status === 'failed' ? 'Retry Stage' : 'Run Stage' }}
      </el-button>
    </div>

    <div v-if="stage.status === 'running'" class="stage-running">
      <el-alert type="info" :closable="false" show-icon>
        <template #title>
          Stage is currently running... Please wait.
        </template>
      </el-alert>
      <el-progress :percentage="100" :indeterminate="true" :stroke-width="6" style="margin-top: 12px" />
    </div>

    <!-- Log viewer -->
    <StageLogViewer
      :session-id="sessionId"
      :stage-id="stage.id"
      :is-running="stage.status === 'running'"
    />

    <!-- Result summary -->
    <div v-if="stage.status === 'completed'" class="stage-result">
      <el-alert :title="stage.summary || 'Stage completed'" type="success" :closable="false" show-icon />
    </div>

    <div v-if="stage.status === 'failed'" class="stage-error">
      <el-alert
        :title="stage.summary || 'Stage failed'"
        type="error"
        :closable="false"
        show-icon
      />
    </div>

    <!-- Intermediate Results — shown after stage completes -->
    <div v-if="stage.status === 'completed'" class="intermediate-results">
      <el-divider />
      <div v-if="!props.hideFileTree" class="results-header">
        <el-icon><FolderOpened /></el-icon>
        <span>Stage Output Files</span>
        <el-button size="small" text @click="showFiles = !showFiles">
          {{ showFiles ? 'Hide' : 'Show' }} Files
        </el-button>
      </div>

      <div v-if="!props.hideFileTree && showFiles" class="results-pane">
        <div class="file-tree-panel">
          <el-tree
            :data="fileTreeData"
            :props="{ label: 'name', children: 'children' }"
            node-key="path"
            highlight-current
            @node-click="onFileClick"
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
        <div class="code-viewer-panel">
          <CodeViewer
            v-if="selectedFile"
            :content="selectedFile.content"
            :language="selectedFile.language"
            :filename="selectedFile.path"
          />
          <div v-else class="no-selection">
            <el-empty description="Select a file to preview" :image-size="80" />
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, watch, nextTick, onMounted } from 'vue'
import { VideoPlay, FolderOpened, Folder, Document } from '@element-plus/icons-vue'
import StatusBadge from '@/components/common/StatusBadge.vue'
import StageLogViewer from '@/components/pipeline/StageLogViewer.vue'
import CodeViewer from '@/components/repo/CodeViewer.vue'
import { getWorkspaceTree, getWorkspaceFile } from '@/apis/index.js'

const props = defineProps({
  stage: { type: Object, required: true },
  sessionId: { type: String, required: true },
  isRunning: { type: Boolean, default: false },
  workspaceSubdir: { type: String, default: '' },
  hideFileTree: { type: Boolean, default: false },
})

defineEmits(['run'])

// For completed stages keep files visible even after page navigation
const showFiles = ref(props.stage.status === 'completed')
const fileTreeData = ref([])
const selectedFile = ref(null)
const loadedStageId = ref('')

// Load file tree on mount if stage already completed (handles re-mount
// after navigating to Result page and back).
onMounted(async () => {
  if (props.stage.status === 'completed') {
    await loadFileTree()
  }
})

// Watch for stage completion → auto-load file tree
watch(() => props.stage.status, async (status) => {
  if (status === 'completed' && props.stage.id !== loadedStageId.value) {
    loadedStageId.value = props.stage.id
    await loadFileTree()
    showFiles.value = true
  }
})

async function loadFileTree() {
  const subdir = props.workspaceSubdir || getDefaultSubdir(props.stage.id)
  try {
    const res = await getWorkspaceTree(props.sessionId, subdir)
    fileTreeData.value = res.data?.data?.file_tree || []
  } catch (e) {
    console.error('Failed to load workspace tree:', e)
    fileTreeData.value = []
  }
}

function getDefaultSubdir(stageId) {
  // Default workspace subdirectories for each stage
  const defaults = {
    stage1_prep: 'skeletons',
    stage2_rag: 'source_skeletons',
    stage3_translate: '',  // root workspace — shows translated, final_projects, etc.
    postprocess: '',        // root output
  }
  return defaults[stageId] || ''
}

async function onFileClick(node) {
  if (node.children !== undefined) return  // folder — ignore
  try {
    const res = await getWorkspaceFile(props.sessionId, node.path)
    selectedFile.value = res.data?.data
  } catch (e) {
    console.error('Failed to load file:', e)
  }
}
</script>

<style scoped>
.stage-panel {
  padding: 8px 0;
}
.stage-title-row {
  display: flex;
  align-items: center;
  gap: 10px;
}
.stage-title-row h4 {
  margin: 0;
  font-size: 16px;
}
.stage-desc {
  margin: 8px 0 0 32px;
  font-size: 13px;
  color: #909399;
}
.stage-actions {
  margin: 16px 0;
  text-align: center;
}
.stage-running {
  margin: 16px 0;
}
.stage-result, .stage-error {
  margin-top: 16px;
}

/* Intermediate results */
.intermediate-results {
  margin-top: 12px;
}
.results-header {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 14px;
  font-weight: 600;
  color: #303133;
}
.results-header .el-button {
  margin-left: auto;
}
.results-pane {
  display: flex;
  margin-top: 12px;
  height: 400px;
  border: 1px solid #ebeef5;
  border-radius: 8px;
  overflow: hidden;
}
.file-tree-panel {
  width: 280px;
  flex-shrink: 0;
  overflow-y: auto;
  border-right: 1px solid #ebeef5;
  padding: 8px;
  background: #fafafa;
}
.file-tree-panel :deep(.el-tree-node__content) {
  height: 28px;
}
.tree-node {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 13px;
}
.code-viewer-panel {
  flex: 1;
  overflow: hidden;
}
.no-selection {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
}
</style>
