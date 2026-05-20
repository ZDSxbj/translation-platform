<template>
  <header class="app-header">
    <div class="header-left" @click="$router.push('/')" style="cursor: pointer">
      <el-icon :size="24" color="#409eff"><Monitor /></el-icon>
      <span class="header-title">Code Translation Platform</span>
    </div>
    <div class="header-right">
      <el-tag v-if="currentRoute" size="small" effect="plain">
        {{ currentRoute }}
      </el-tag>
    </div>
  </header>
</template>

<script setup>
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import { Monitor } from '@element-plus/icons-vue'

const route = useRoute()

const currentRoute = computed(() => {
  if (route.path === '/') return 'Home'
  if (route.path.startsWith('/workspace')) return 'Pipeline'
  if (route.path.startsWith('/result')) return 'Results'
  return null
})
</script>

<style scoped>
.app-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: 56px;
  padding: 0 24px;
  background: #fff;
  border-bottom: 1px solid #e4e7ed;
  box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.header-left {
  display: flex;
  align-items: center;
  gap: 10px;
}
.header-title {
  font-size: 17px;
  font-weight: 600;
  color: #303133;
}
</style>
