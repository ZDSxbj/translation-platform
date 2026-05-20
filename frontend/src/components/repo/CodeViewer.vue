<template>
  <div class="code-viewer">
    <div class="code-viewer-header">
      <span class="code-filename">{{ filename || 'Untitled' }}</span>
      <el-tag v-if="language" size="small" type="info">{{ language }}</el-tag>
    </div>
    <div ref="editorContainer" class="code-editor-container"></div>
  </div>
</template>

<script setup>
import { ref, onMounted, watch, nextTick, onBeforeUnmount } from 'vue'
import { EditorView, basicSetup } from 'codemirror'
import { EditorState } from '@codemirror/state'
import { cpp } from '@codemirror/lang-cpp'
import { rust } from '@codemirror/lang-rust'
import { oneDark } from '@codemirror/theme-one-dark'

const props = defineProps({
  content: { type: String, default: '' },
  language: { type: String, default: 'text' },
  filename: { type: String, default: '' },
})

const editorContainer = ref(null)
let editorView = null

function getLanguageExtension(lang) {
  const map = {
    c: cpp, cpp: cpp, h: cpp, hpp: cpp,
    rust: rust, rs: rust,
  }
  return map[lang] ? map[lang]() : null
}

onMounted(() => {
  createEditor()
})

watch(() => props.content, (val) => {
  if (editorView && val !== editorView.state.doc.toString()) {
    editorView.dispatch({
      changes: { from: 0, to: editorView.state.doc.length, insert: val },
    })
  }
})

watch(() => props.language, () => {
  // Re-create editor when language changes
  if (editorView) {
    editorView.destroy()
  }
  nextTick(() => createEditor())
})

onBeforeUnmount(() => {
  if (editorView) {
    editorView.destroy()
    editorView = null
  }
})

function createEditor() {
  if (!editorContainer.value) return

  const langExt = getLanguageExtension(props.language)
  const extensions = [
    basicSetup,
    EditorView.editable.of(false),
    oneDark,
  ]
  if (langExt) extensions.push(langExt)

  editorView = new EditorView({
    state: EditorState.create({
      doc: props.content,
      extensions,
    }),
    parent: editorContainer.value,
  })
}
</script>

<style scoped>
.code-viewer {
  display: flex;
  flex-direction: column;
  height: 100%;
}
.code-viewer-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 16px;
  background: #252526;
  border-bottom: 1px solid #3e3e42;
}
.code-filename {
  font-size: 13px;
  color: #cccccc;
  font-family: 'Consolas', 'Monaco', monospace;
}
.code-editor-container {
  flex: 1;
  overflow: auto;
}
</style>
