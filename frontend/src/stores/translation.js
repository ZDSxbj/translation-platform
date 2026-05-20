import { defineStore } from 'pinia'
import {
  startTranslation, getSessionState, runStage,
  getStageResult, getStageLogs, getOutputTree, getOutputFile,
} from '@/apis/index.js'

export const useTranslationStore = defineStore('translation', {
  state: () => ({
    projectId: null,
    sessionId: null,
    config: {},
    stages: [],
    currentStageIndex: 0,
    isRunning: false,
    error: null,
  }),

  getters: {
    currentStageId(state) {
      return state.stages[state.currentStageIndex]?.id || null
    },
    allStagesComplete(state) {
      return state.stages.length > 0 &&
        state.stages.every((s) => s.status === 'completed' || s.status === 'skipped')
    },
    hasFailedStage(state) {
      return state.stages.some((s) => s.status === 'failed')
    },
  },

  actions: {
    async initSession(projectId, config) {
      this.projectId = projectId
      this.config = config

      const res = await startTranslation(projectId, config)
      const data = res.data?.data
      this.sessionId = data.session_id
      this.stages = data.stages.map((s) => ({
        id: s.id,
        name: s.name,
        status: s.status || 'pending',
        summary: '',
      }))
      this.currentStageIndex = 0
      this.error = null
    },

    async refreshState() {
      if (!this.sessionId) return
      const res = await getSessionState(this.sessionId)
      const data = res.data?.data
      if (data?.stages) {
        this.stages = data.stages
        this.currentStageIndex = data.current_stage_index || 0
      }
    },

    async runStageAction(stageId) {
      if (!this.sessionId) return null
      this.isRunning = true
      this.error = null

      // Update local stage status
      const stage = this.stages.find((s) => s.id === stageId)
      if (stage) stage.status = 'running'

      try {
        const res = await runStage(this.sessionId, stageId)
        const data = res.data?.data

        // Update stage status from response
        if (data) {
          if (stage) {
            stage.status = data.status
            stage.summary = data.summary || ''
          }
          // Update current stage index
          if (data.status === 'completed') {
            const idx = this.stages.findIndex((s) => s.id === stageId)
            if (idx >= 0 && idx === this.currentStageIndex) {
              this.currentStageIndex = idx + 1
            }
          }
        }

        this.isRunning = false
        return data
      } catch (e) {
        if (stage) stage.status = 'failed'
        this.error = e.response?.data?.message || e.message
        this.isRunning = false
        throw e
      }
    },

    async getStageResult(stageId) {
      if (!this.sessionId) return null
      const res = await getStageResult(this.sessionId, stageId)
      return res.data?.data
    },

    async getStageLogs(stageId) {
      if (!this.sessionId) return []
      const res = await getStageLogs(this.sessionId, stageId)
      return res.data?.data || []
    },

    async getOutputTree(subdir = '') {
      if (!this.sessionId) return []
      const res = await getOutputTree(this.sessionId, subdir)
      return res.data?.data?.file_tree || []
    },

    async getOutputFileContent(path) {
      if (!this.sessionId) return null
      const res = await getOutputFile(this.sessionId, path)
      return res.data?.data
    },
  },
})
