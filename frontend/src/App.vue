<template>
  <div id="app-root">
    <AppHeader />
    <main class="app-main">
      <router-view />
    </main>
  </div>
</template>

<script setup>
import AppHeader from '@/components/layout/AppHeader.vue'
</script>

<style scoped>
.app-main {
  min-height: calc(100vh - 60px);
  background-color: #f5f7fa;
}
</style>
