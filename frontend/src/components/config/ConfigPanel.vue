<template>
  <div class="config-panel">
    <el-form :model="localConfig" label-position="top" :disabled="disabled">
      <!-- Engine Selection -->
      <el-form-item label="Translation Engine">
        <el-select v-model="localConfig.engine" style="width: 100%">
          <el-option label="His2Trans (C/C++ → Rust)" value="his2trans" />
        </el-select>
      </el-form-item>

      <!-- LLM Model -->
      <el-form-item label="LLM Model">
        <el-input v-model="localConfig.model" placeholder="e.g., deepseek-v3.2" />
        <div class="form-hint">OpenAI-compatible model name. Examples: deepseek-v3.2, deepseek-coder, gpt-4o</div>
      </el-form-item>

      <!-- OpenHarmony Source Root (OHOS projects only) -->
      <el-form-item
        v-if="projectType === 'ohos'"
        label="OpenHarmony Source Root"
      >
        <el-input
          v-model="localConfig.ohos_root"
          placeholder="/path/to/ohos/source/tree or ohos_root_min/"
        />
        <div class="form-hint">
          Required for resolving OpenHarmony SDK headers (hdf_*, hilog, etc.).
          This should point to the root of the OpenHarmony source tree or <code>ohos_root_min/</code> folder.
        </div>
      </el-form-item>

      <!-- Additional Include Directories -->
      <el-form-item label="Additional Include Directories">
        <div class="includes-list">
          <div v-for="(inc, idx) in localConfig.extra_includes" :key="idx" class="include-row">
            <el-input
              v-model="localConfig.extra_includes[idx]"
              placeholder="/path/to/include"
              size="small"
            />
            <el-button
              :icon="Delete"
              circle
              size="small"
              type="danger"
              @click="removeInclude(idx)"
            />
          </div>
          <el-button
            size="small"
            :icon="Plus"
            @click="addInclude"
          >
            Add Include Path
          </el-button>
        </div>
        <div class="form-hint">Additional directories to search for header files during compilation</div>
      </el-form-item>

      <!-- RAG Toggle -->
      <el-form-item label="RAG (Retrieval-Augmented Generation)">
        <el-switch v-model="localConfig.use_rag" active-text="On" inactive-text="Off" />
        <div class="form-hint">Enable BM25 + Jina Reranker for signature matching with knowledge base</div>
      </el-form-item>

      <!-- Max Repair Rounds -->
      <el-form-item label="Max Repair Rounds">
        <el-input-number
          v-model="localConfig.max_repair"
          :min="0"
          :max="10"
          :step="1"
          style="width: 100%"
        />
        <div class="form-hint">Maximum compile-and-repair iterations per function (0 = no repair)</div>
      </el-form-item>

      <!-- Advanced Settings -->
      <el-collapse>
        <el-collapse-item title="Advanced Settings" name="advanced">
          <el-form-item label="API Base URL">
            <el-input v-model="localConfig.api_base_url" placeholder="https://api.apiyi.com/v1" />
          </el-form-item>
          <el-form-item label="Max Tokens">
            <el-input-number v-model="localConfig.api_max_tokens" :min="256" :max="65536" :step="256" style="width: 100%" />
          </el-form-item>
          <el-form-item label="Temperature">
            <el-slider v-model="localConfig.api_temperature" :min="0" :max="2" :step="0.1" show-input />
          </el-form-item>
          <el-form-item label="API Key">
            <el-input
              v-model="localConfig.api_key"
              placeholder="Use env default if empty"
              type="password"
              show-password
            />
          </el-form-item>
        </el-collapse-item>
      </el-collapse>
    </el-form>
  </div>
</template>

<script setup>
import { computed, watch } from 'vue'
import { Plus, Delete } from '@element-plus/icons-vue'

const props = defineProps({
  modelValue: { type: Object, required: true },
  disabled: { type: Boolean, default: false },
  projectType: { type: String, default: null },
})

const emit = defineEmits(['update:modelValue'])

const localConfig = computed({
  get: () => props.modelValue,
  set: (val) => emit('update:modelValue', val),
})

function addInclude() {
  const updated = { ...localConfig.value }
  if (!updated.extra_includes) updated.extra_includes = []
  updated.extra_includes.push('')
  emit('update:modelValue', updated)
}

function removeInclude(idx) {
  const updated = { ...localConfig.value }
  if (updated.extra_includes) {
    updated.extra_includes.splice(idx, 1)
  }
  emit('update:modelValue', updated)
}

// Ensure arrays are initialized
watch(() => props.modelValue, (val) => {
  if (val && !val.extra_includes) {
    emit('update:modelValue', { ...val, extra_includes: [] })
  }
}, { immediate: true })
</script>

<style scoped>
.config-panel {
  width: 100%;
}
.form-hint {
  font-size: 12px;
  color: #909399;
  margin-top: 2px;
  line-height: 1.4;
}
.form-hint code {
  background: #f0f2f5;
  padding: 1px 4px;
  border-radius: 3px;
  font-size: 11px;
}
.includes-list {
  width: 100%;
}
.include-row {
  display: flex;
  gap: 6px;
  margin-bottom: 6px;
}
.include-row .el-input {
  flex: 1;
}
</style>
