<template>
  <div class="stage-log-viewer">
    <div class="log-header" @click="expanded = !expanded">
      <el-icon><component :is="expanded ? ArrowDown : ArrowRight" /></el-icon>
      <span>Log Output</span>
      <el-tag v-if="logs.length > 0" size="small" type="info">{{ logs.length }} entries</el-tag>
      <el-button v-if="logs.length > 0" link size="small" @click.stop="refreshLogs">Refresh</el-button>
    </div>

    <div v-if="expanded" class="log-body" ref="logBody">
      <div v-if="logs.length === 0 && !isRunning" class="log-empty">
        No log entries yet.
      </div>
      <div v-for="(entry, i) in logs" :key="i" class="log-entry" :class="`log-${entry.level}`">
        <span class="log-time">{{ formatTime(entry.timestamp) }}</span>
        <span class="log-msg">{{ entry.message }}</span>
      </div>
      <div v-if="isRunning" class="log-waiting">
        <el-icon class="is-loading"><Loading /></el-icon>
        Waiting for output...
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, watch, nextTick, onMounted } from 'vue'
import { ArrowDown, ArrowRight, Loading } from '@element-plus/icons-vue'
import { getStageLogs } from '@/apis/index.js'

const props = defineProps({
  sessionId: { type: String, required: true },
  stageId: { type: String, required: true },
  isRunning: { type: Boolean, default: false },
})

const expanded = ref(false)
const logs = ref([])
const logBody = ref(null)
let pollTimer = null

onMounted(() => {
  if (props.isRunning) {
    expanded.value = true
    startPolling()
  }
})

watch(() => props.isRunning, (running) => {
  if (running) {
    expanded.value = true
    startPolling()
  } else {
    stopPolling()
    refreshLogs()
  }
})

async function refreshLogs() {
  try {
    const res = await getStageLogs(props.sessionId, props.stageId)
    logs.value = res.data?.data || []
    await nextTick()
    scrollToBottom()
  } catch (e) {
    // ignore
  }
}

function startPolling() {
  stopPolling()
  refreshLogs()
  pollTimer = setInterval(refreshLogs, 2000)
}

function stopPolling() {
  if (pollTimer) {
    clearInterval(pollTimer)
    pollTimer = null
  }
}

function scrollToBottom() {
  if (logBody.value) {
    logBody.value.scrollTop = logBody.value.scrollHeight
  }
}

function formatTime(ts) {
  if (!ts) return ''
  try {
    const d = new Date(ts)
    return d.toLocaleTimeString()
  } catch {
    return ''
  }
}
</script>

<style scoped>
.stage-log-viewer {
  margin-top: 16px;
  border: 1px solid #ebeef5;
  border-radius: 8px;
  overflow: hidden;
}
.log-header {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  background: #fafafa;
  cursor: pointer;
  font-size: 13px;
  font-weight: 500;
}
.log-body {
  max-height: 300px;
  overflow-y: auto;
  padding: 8px 12px;
  background: #1e1e1e;
  font-family: 'Consolas', 'Monaco', monospace;
  font-size: 12px;
  line-height: 1.6;
}
.log-empty {
  color: #888;
  font-style: italic;
}
.log-entry {
  display: flex;
  gap: 8px;
}
.log-time {
  color: #888;
  flex-shrink: 0;
}
.log-msg {
  color: #d4d4d4;
  word-break: break-all;
}
.log-info .log-msg { color: #d4d4d4; }
.log-warn .log-msg { color: #e6a23c; }
.log-error .log-msg { color: #f56c6c; }
.log-waiting {
  color: #888;
  display: flex;
  align-items: center;
  gap: 6px;
}
</style>
