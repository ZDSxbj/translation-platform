<template>
  <div class="pipeline-stepper">
    <el-steps :active="activeStep" finish-status="success" align-center>
      <template v-for="(stage, index) in stages" :key="stage.id">
        <el-step
          :title="stage.name.split(':')[0]"
          :description="stage.description || stage.summary || statusLabel(stage.status)"
          :status="stepStatus(stage.status)"
        />
        <div v-if="index < stages.length - 1" class="step-connector">
          <el-divider direction="vertical" />
        </div>
      </template>
    </el-steps>

    <!-- Stage status legend -->
    <div class="stage-legend">
      <div v-for="stage in stages" :key="stage.id" class="legend-item">
        <StatusBadge :status="stage.status" />
        <span class="legend-name">{{ stage.name.split(':')[0] }}</span>
        <span v-if="stage.summary" class="legend-summary">— {{ stage.summary }}</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import StatusBadge from '@/components/common/StatusBadge.vue'

const props = defineProps({
  stages: { type: Array, required: true },
  currentIndex: { type: Number, default: 0 },
})

const activeStep = computed(() => {
  // Find the index of the first non-completed stage
  for (let i = 0; i < props.stages.length; i++) {
    if (props.stages[i].status !== 'completed' && props.stages[i].status !== 'skipped') {
      return i
    }
  }
  return props.stages.length
})

function stepStatus(stageStatus) {
  const map = {
    pending: 'wait',
    running: 'process',
    completed: 'success',
    failed: 'error',
    skipped: 'success',
  }
  return map[stageStatus] || 'wait'
}

function statusLabel(status) {
  const map = {
    pending: 'Pending',
    running: 'Running...',
    completed: 'Completed',
    failed: 'Failed',
    skipped: 'Skipped',
  }
  return map[status] || ''
}
</script>

<style scoped>
.pipeline-stepper {
  padding: 4px 0;
}
.stage-legend {
  display: flex;
  justify-content: center;
  gap: 24px;
  margin-top: 20px;
  flex-wrap: wrap;
}
.legend-item {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 13px;
}
.legend-name {
  font-weight: 500;
  color: #606266;
}
.legend-summary {
  color: #909399;
  max-width: 200px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
</style>
