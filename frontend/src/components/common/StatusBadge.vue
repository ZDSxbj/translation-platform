<template>
  <span class="status-badge" :class="`status-${status}`">
    <el-icon v-if="status === 'running'" class="is-loading"><Loading /></el-icon>
    <el-icon v-else-if="status === 'completed'"><CircleCheckFilled /></el-icon>
    <el-icon v-else-if="status === 'failed'"><CircleCloseFilled /></el-icon>
    <el-icon v-else-if="status === 'skipped'"><RemoveFilled /></el-icon>
    <el-icon v-else><Clock /></el-icon>
  </span>
</template>

<script setup>
import {
  Loading, CircleCheckFilled, CircleCloseFilled, RemoveFilled, Clock,
} from '@element-plus/icons-vue'

defineProps({
  status: { type: String, default: 'pending' },
})
</script>

<style scoped>
.status-badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 22px;
  height: 22px;
  border-radius: 50%;
}
.status-pending { color: #c0c4cc; }
.status-running { color: #409eff; }
.status-completed { color: #67c23a; }
.status-failed { color: #f56c6c; }
.status-skipped { color: #909399; }
</style>
