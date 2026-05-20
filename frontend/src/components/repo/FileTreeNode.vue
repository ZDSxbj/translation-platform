<template>
  <div class="tree-node">
    <!-- Directory -->
    <div
      v-if="node.type === 'directory'"
      class="node-row"
      :style="{ paddingLeft: depth * 16 + 'px' }"
    >
      <span class="node-toggle" @click="expanded = !expanded">
        <el-icon :size="14">
          <component :is="expanded ? FolderOpened : Folder" />
        </el-icon>
      </span>
      <span class="node-name" @click="expanded = !expanded">{{ node.name }}</span>
      <span v-if="node.children" class="node-count">{{ node.children.length }}</span>
    </div>

    <!-- Children (recursive) -->
    <template v-if="node.type === 'directory' && expanded">
      <FileTreeNode
        v-for="child in node.children"
        :key="child.path"
        :node="child"
        :depth="depth + 1"
        :selected-path="selectedPath"
        @file-selected="$emit('file-selected', $event)"
      />
    </template>

    <!-- File -->
    <div
      v-if="node.type === 'file'"
      class="node-row file-row"
      :class="{ selected: selectedPath === node.path }"
      :style="{ paddingLeft: depth * 16 + 'px' }"
      @click="onSelect"
    >
      <span class="node-toggle">
        <el-icon :size="14">
          <component :is="fileIcon" />
        </el-icon>
      </span>
      <span class="node-name">{{ node.name }}</span>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, markRaw } from 'vue'
import {
  Folder, FolderOpened, Document, Tickets,
} from '@element-plus/icons-vue'

const props = defineProps({
  node: { type: Object, required: true },
  depth: { type: Number, default: 0 },
  selectedPath: { type: String, default: null },
})

const emit = defineEmits(['file-selected'])

const expanded = ref(false)

const fileIcon = computed(() => {
  const lang = props.node.language
  if (lang === 'rust') return markRaw(Tickets)
  if (lang === 'c' || lang === 'cpp') return markRaw(Tickets)
  return markRaw(Document)
})

function onSelect() {
  emit('file-selected', props.node)
}
</script>

<style scoped>
.node-row {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 4px 8px;
  cursor: pointer;
  border-radius: 4px;
  margin: 1px 4px;
  transition: background 0.15s;
}
.node-row:hover {
  background: #f0f2f5;
}
.file-row.selected {
  background: #ecf5ff;
  color: #409eff;
}
.node-toggle {
  width: 20px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  color: #909399;
}
.node-name {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.node-count {
  font-size: 11px;
  color: #c0c4cc;
}
</style>
