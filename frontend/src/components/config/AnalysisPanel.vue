<template>
  <div v-if="analysis" class="analysis-panel">
    <!-- Project Type Badge -->
    <div class="analyze-header">
      <div class="header-left">
        <span class="label">Project Type</span>
        <el-tag
          :type="typeBadgeType"
          size="large"
          effect="dark"
          round
        >
          <el-icon style="margin-right: 4px">
            <component :is="typeIcon" />
          </el-icon>
          {{ typeLabel }}
        </el-tag>
      </div>
      <div class="header-right">
        <el-tag v-if="analysis.has_compile_commands" type="success" size="small" effect="plain">
          compile_commands.json found
        </el-tag>
        <el-tag v-else type="info" size="small" effect="plain">
          no compile_commands.json
        </el-tag>
      </div>
    </div>

    <!-- Compile Commands Info -->
    <div v-if="analysis.has_compile_commands && analysis.compile_commands_info" class="info-grid">
      <div class="info-item">
        <span class="info-label">Entries</span>
        <span class="info-value">{{ analysis.compile_commands_info.entry_count }}</span>
      </div>
      <div class="info-item">
        <span class="info-label">Source Files</span>
        <span class="info-value">{{ analysis.compile_commands_info.source_files_in_db?.length || 0 }}</span>
      </div>
      <div class="info-item">
        <span class="info-label">Include Dirs</span>
        <span class="info-value">{{ analysis.compile_commands_info.include_dirs?.length || 0 }}</span>
      </div>
      <div class="info-item">
        <span class="info-label">External Includes</span>
        <span class="info-value" :class="{ 'text-warning': analysis.compile_commands_info.external_includes?.length }">
          {{ analysis.compile_commands_info.external_includes?.length || 0 }}
        </span>
      </div>
    </div>

    <!-- Broken Paths Warning -->
    <div v-if="brokenPaths.length > 0" class="warning-box">
      <el-alert
        :title="`${brokenPaths.length} unresolvable include path(s) detected`"
        type="warning"
        :closable="false"
        show-icon
      >
        <template #default>
          <div class="broken-path-list">
            <el-tag
              v-for="path in brokenPaths.slice(0, 6)"
              :key="path"
              size="small"
              type="warning"
              effect="plain"
              class="broken-tag"
            >{{ path }}</el-tag>
            <span v-if="brokenPaths.length > 6" class="more-hint">
              +{{ brokenPaths.length - 6 }} more
            </span>
          </div>
        </template>
      </el-alert>
    </div>

    <!-- Absolute Paths Warning -->
    <div v-if="needsPathFixup" class="fixup-box">
      <el-alert
        title="Compile commands contain absolute paths from the build machine"
        description="These paths likely won't resolve on this server. Click 'Relativize Paths' to rewrite them relative to the project root."
        type="info"
        :closable="false"
        show-icon
      />
      <el-button
        type="primary"
        size="small"
        :loading="fixing"
        :icon="RefreshRight"
        @click="onFixPaths"
        style="margin-top: 8px"
      >
        {{ fixing ? 'Relativizing...' : 'Relativize Paths' }}
      </el-button>
    </div>

    <!-- SDK Requirements -->
    <div v-if="needsOhosRoot" class="sdk-box">
      <el-alert
        title="OpenHarmony SDK Required"
        :description="sdkDescription"
        type="info"
        :closable="false"
        show-icon
      />
      <el-form-item label="OpenHarmony Source Root" style="margin-top: 12px">
        <el-input
          :model-value="ohosRoot"
          placeholder="/path/to/ohos/source/tree or ohos_root_min/"
          @update:model-value="$emit('update:ohosRoot', $event)"
        >
          <template #prepend>
            <el-icon><FolderOpened /></el-icon>
          </template>
        </el-input>
        <div class="form-hint">
          Path to the OpenHarmony source tree (e.g., <code>ohos_root_min/</code> in the project ZIP).
          Required to resolve headers like <code>hdf_device_desc.h</code>.
        </div>
      </el-form-item>
    </div>

    <!-- Standard C — All Clear -->
    <div v-if="isStandardC" class="ok-box">
      <el-alert
        title="Standard C project — no external dependencies detected"
        description="All includes appear to be from the standard C library. No special configuration needed."
        type="success"
        :closable="false"
        show-icon
      />
    </div>

    <!-- Recommendation Summary -->
    <div class="recommendation-bar">
      <el-icon><InfoFilled /></el-icon>
      <span>{{ recommendationText }}</span>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch } from 'vue'
import { ElMessage } from 'element-plus'
import {
  FolderOpened, InfoFilled, RefreshRight,
  Monitor, SetUp, QuestionFilled, CircleCheck,
} from '@element-plus/icons-vue'
import { analyzeProject, fixProjectPaths } from '@/apis/index.js'

const props = defineProps({
  projectId: { type: String, default: null },
  modelValue: { type: Object, default: () => ({}) },
})

const emit = defineEmits(['update:modelValue', 'analysis-complete', 'update:ohosRoot'])

const ohosRoot = computed({
  get: () => props.modelValue.ohos_root || '',
  set: (val) => emit('update:modelValue', { ...props.modelValue, ohos_root: val }),
})

const analysis = ref(null)
const fixing = ref(false)

// Derived
const projectType = computed(() => analysis.value?.project_type || 'unknown')
const isStandardC = computed(() => projectType.value === 'standard_c')
const isOhos = computed(() => projectType.value === 'ohos')
const isUnknown = computed(() => projectType.value === 'unknown')

const typeBadgeType = computed(() => {
  if (isOhos.value) return 'warning'
  if (isStandardC.value) return 'success'
  return 'info'
})

const typeLabel = computed(() => {
  if (isOhos.value) return 'OHOS / Complex Build'
  if (isStandardC.value) return 'Standard C'
  return 'Unknown'
})

const typeIcon = computed(() => {
  if (isOhos.value) return SetUp
  if (isStandardC.value) return CircleCheck
  return QuestionFilled
})

const brokenPaths = computed(() => {
  return analysis.value?.compile_commands_info?.broken_paths || []
})

const needsPathFixup = computed(() => {
  return analysis.value?.recommendation?.path_fixup_needed || false
})

const needsOhosRoot = computed(() => {
  return analysis.value?.recommendation?.needs_openharmony_root || false
})

const sdkDescription = computed(() => {
  const sdks = analysis.value?.detected_dependencies?.external_sdks || []
  if (sdks.length === 0) return 'This project uses external headers that cannot be resolved with standard libraries alone.'
  return `Detected SDK dependencies: ${sdks.join(', ')}. Requires OpenHarmony SDK headers.`
})

const recommendationText = computed(() => {
  const rec = analysis.value?.recommendation || {}
  if (rec.can_auto_compile) {
    return 'This project can be compiled automatically. No additional path configuration needed.'
  }
  if (rec.needs_openharmony_root) {
    return 'Please specify the OpenHarmony SDK root path below, then fix any absolute paths.'
  }
  if (rec.path_fixup_needed) {
    return 'Absolute paths detected. Click "Relativize Paths" to fix before starting translation.'
  }
  return 'Project analysis complete. Review the details above before proceeding.'
})

// Watch projectId to trigger analysis
watch(() => props.projectId, (newId) => {
  if (newId) {
    onLoadAnalysis()
  }
}, { immediate: true })

async function onLoadAnalysis() {
  if (!props.projectId) return
  try {
    const res = await analyzeProject(props.projectId)
    analysis.value = res.data?.data
    emit('analysis-complete', analysis.value)
  } catch (e) {
    ElMessage.warning('Project analysis failed. You can still proceed with manual configuration.')
  }
}

async function onFixPaths() {
  if (!props.projectId) return
  fixing.value = true
  try {
    const res = await fixProjectPaths(props.projectId)
    if (res.data?.code === 200) {
      ElMessage.success(res.data?.data?.message || 'Paths relativized')
      // Re-run analysis to refresh broken path list
      await onLoadAnalysis()
    } else {
      ElMessage.error(res.data?.message || 'Failed to fix paths')
    }
  } catch (e) {
    ElMessage.error('Failed to relativize paths')
  } finally {
    fixing.value = false
  }
}
</script>

<style scoped>
.analysis-panel {
  width: 100%;
}
.analyze-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}
.header-left {
  display: flex;
  align-items: center;
  gap: 8px;
}
.header-left .label {
  font-size: 13px;
  color: #909399;
}
.info-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
  margin-bottom: 16px;
}
.info-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 6px 10px;
  background: #f5f7fa;
  border-radius: 6px;
}
.info-label {
  font-size: 12px;
  color: #909399;
}
.info-value {
  font-size: 14px;
  font-weight: 600;
  color: #303133;
}
.info-value.text-warning {
  color: #e6a23c;
}
.warning-box, .fixup-box, .sdk-box, .ok-box {
  margin-bottom: 12px;
}
.broken-path-list {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  margin-top: 6px;
}
.broken-tag {
  font-family: monospace;
  font-size: 11px;
}
.more-hint {
  font-size: 12px;
  color: #909399;
  align-self: center;
}
.form-hint {
  font-size: 11px;
  color: #909399;
  margin-top: 4px;
  line-height: 1.4;
}
.form-hint code {
  background: #f0f2f5;
  padding: 1px 4px;
  border-radius: 3px;
  font-size: 11px;
}
.recommendation-bar {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 12px;
  background: #ecf5ff;
  border-radius: 8px;
  font-size: 13px;
  color: #409eff;
  margin-top: 12px;
}
</style>
