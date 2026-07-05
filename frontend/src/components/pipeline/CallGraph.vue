<template>
  <div class="call-graph-wrapper">
    <div ref="graphContainer" class="graph-canvas"></div>
  </div>
</template>

<script setup>
import { ref, onMounted, watch, nextTick } from 'vue'
import { Network } from 'vis-network'
import { DataSet } from 'vis-data'

const props = defineProps({
  nodes: { type: Array, default: () => [] },
  edges: { type: Array, default: () => [] },
})

const graphContainer = ref(null)
let network = null

function buildGraph() {
  if (!graphContainer.value || !props.nodes.length) return

  const nodesArr = props.nodes.map(n => ({
    id: n.id || n.label,
    label: n.label,
    color: {
      background: n.is_external ? '#fff3e0' : '#e8f5e9',
      border: n.is_external ? '#e65100' : '#2e7d32',
    },
    font: { size: 13, face: 'JetBrains Mono, monospace' },
    borderWidth: 1.5,
    shape: 'box',
    margin: 8,
  }))

  const edgesArr = props.edges.map((e, i) => ({
    id: `${e.source}_${e.target}_${i}`,
    from: e.source,
    to: e.target,
    color: { color: e.is_external ? '#e65100' : '#2e7d32', opacity: 0.6 },
    arrows: 'to',
    width: 1.2,
  }))

  const data = {
    nodes: new DataSet(nodesArr),
    edges: new DataSet(edgesArr),
  }

  const options = {
    layout: {
      hierarchical: {
        direction: 'UD',
        sortMethod: 'directed',
        nodeSpacing: 150,
        levelSeparation: 120,
      },
    },
    physics: {
      hierarchicalRepulsion: { nodeDistance: 140 },
    },
    interaction: {
      hover: true,
      tooltipDelay: 200,
    },
    edges: {
      smooth: { type: 'curvedCW', roundness: 0.2 },
    },
  }

  if (network) {
    network.setData(data)
  } else {
    network = new Network(graphContainer.value, data, options)
  }
}

onMounted(() => { nextTick(buildGraph) })
watch(() => [props.nodes, props.edges], () => { nextTick(buildGraph) }, { deep: true })
</script>

<style scoped>
.call-graph-wrapper { width: 100%; }
.graph-canvas { width: 100%; height: 500px; border: 1px solid #ebeef5; border-radius: 8px; background: #fafbfc; }
</style>
