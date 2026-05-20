import { createRouter, createWebHistory } from 'vue-router'
import HomePage from '@/views/HomePage.vue'
import TranslationWorkspace from '@/views/TranslationWorkspace.vue'
import ResultViewer from '@/views/ResultViewer.vue'

const routes = [
  {
    path: '/',
    name: 'Home',
    component: HomePage,
  },
  {
    path: '/workspace/:sessionId',
    name: 'Workspace',
    component: TranslationWorkspace,
    props: true,
  },
  {
    path: '/result/:sessionId',
    name: 'Result',
    component: ResultViewer,
    props: true,
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

export default router
