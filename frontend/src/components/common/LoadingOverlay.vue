<template>
  <div v-if="visible" class="loading-overlay">
    <div class="loading-content">
      <el-icon class="is-loading" :size="32"><Loading /></el-icon>
      <p>{{ text }}</p>
    </div>
  </div>
</template>

<script setup>
import { Loading } from '@element-plus/icons-vue'

defineProps({
  visible: { type: Boolean, default: false },
  text: { type: String, default: 'Loading...' },
})
</script>

<style scoped>
.loading-overlay {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(255,255,255,0.85);
  z-index: 100;
  border-radius: 12px;
}
.loading-content {
  text-align: center;
  color: #409eff;
}
.loading-content p {
  margin: 12px 0 0;
  font-size: 14px;
}
</style>
