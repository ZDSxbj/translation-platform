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

      <StagePanel
        :stage="stage"
        :session-id="store.sessionId"
        :is-running="stage.status === 'running'"
        :workspace-subdir="getWorkspaceSubdir(stage.id)"
        @run="onRunStage"
      />
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
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { ArrowLeft, VideoPlay, Finished } from '@element-plus/icons-vue'
import { useTranslationStore } from '@/stores/translation'
import PipelineStepper from '@/components/pipeline/PipelineStepper.vue'
import StagePanel from '@/components/pipeline/StagePanel.vue'
import StatusBadge from '@/components/common/StatusBadge.vue'

const props = defineProps({
  sessionId: { type: String, required: true },
})

const router = useRouter()
const store = useTranslationStore()

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
  // Show stages 0..idx (completed + current)
  const result = []
  for (let i = 0; i <= Math.min(idx, stages.length - 1); i++) {
    result.push(stages[i])
  }
  // Also show first pending stage if all completed up to a point
  if (result.length === 0 && stages.length > 0) {
    result.push(stages[0])
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
    postprocess: '',
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
</style>
