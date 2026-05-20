<template>
  <div class="zip-uploader">
    <el-upload
      ref="uploadRef"
      class="upload-area"
      drag
      :auto-upload="false"
      :limit="1"
      accept=".zip"
      :disabled="disabled"
      :on-change="onFileChange"
      :on-remove="onFileRemove"
    >
      <el-icon class="upload-icon"><UploadFilled /></el-icon>
      <div class="upload-text">
        <p class="upload-primary">Drop your C/C++ project ZIP here</p>
        <p class="upload-secondary">or click to browse</p>
      </div>
      <template #tip>
        <div class="upload-tip">
          Supported: .zip files containing C/C++ source code (max 500MB).
          Include compile_commands.json for better analysis.
        </div>
      </template>
    </el-upload>

    <div v-if="uploading" class="upload-progress">
      <el-progress :percentage="uploadProgress" :stroke-width="8" />
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { UploadFilled } from '@element-plus/icons-vue'
import { uploadZip } from '@/apis/index.js'

const props = defineProps({
  disabled: { type: Boolean, default: false },
})

const emit = defineEmits(['uploaded', 'error'])

const uploadRef = ref(null)
const uploading = ref(false)
const uploadProgress = ref(0)

async function onFileChange(file) {
  if (!file?.raw) return

  uploading.value = true
  uploadProgress.value = 30

  try {
    const res = await uploadZip(file.raw)
    uploadProgress.value = 100

    if (res.data?.code === 200) {
      emit('uploaded', res.data.data)
    } else {
      emit('error', res.data?.message || 'Upload failed')
    }
  } catch (e) {
    emit('error', e.response?.data?.message || e.message || 'Upload error')
  } finally {
    uploading.value = false
    setTimeout(() => { uploadProgress.value = 0 }, 1000)
  }
}

function onFileRemove() {
  emit('uploaded', null)
}
</script>

<style scoped>
.zip-uploader {
  width: 100%;
}
.upload-area {
  width: 100%;
}
.upload-icon {
  font-size: 48px;
  color: #409eff;
}
.upload-text {
  margin-top: 8px;
}
.upload-primary {
  font-size: 15px;
  color: #303133;
  margin: 0;
}
.upload-secondary {
  font-size: 13px;
  color: #909399;
  margin: 4px 0 0;
}
.upload-tip {
  font-size: 12px;
  color: #909399;
  line-height: 1.5;
}
.upload-progress {
  margin-top: 12px;
}
</style>
