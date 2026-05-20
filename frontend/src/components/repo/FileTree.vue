<template>
  <div class="file-tree">
    <div v-if="tree.length === 0" class="tree-empty">
      <span>No files</span>
    </div>
    <FileTreeNode
      v-for="node in tree"
      :key="node.path"
      :node="node"
      :depth="0"
      :selected-path="selectedPath"
      @file-selected="$emit('file-selected', $event)"
    />
  </div>
</template>

<script setup>
import { ref } from 'vue'
import FileTreeNode from '@/components/repo/FileTreeNode.vue'

defineProps({
  tree: { type: Array, default: () => [] },
})

defineEmits(['file-selected'])

const selectedPath = ref(null)
</script>

<style scoped>
.file-tree {
  padding: 4px 0;
  font-size: 13px;
}
.tree-empty {
  padding: 16px;
  text-align: center;
  color: #909399;
}
</style>
