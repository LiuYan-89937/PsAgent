<script setup lang="ts">
import { computed } from 'vue'
import type { JobDetailResponse } from '@/types/api'
import ExecutionSummary from '@/components/ExecutionSummary.vue'

const props = defineProps<{
  jobDetail: JobDetailResponse
}>()

const inputImage = computed(() => props.jobDetail.input_assets?.[0]?.content_url)
const outputImage = computed(() => props.jobDetail.selected_output?.content_url)
</script>

<template>
  <div class="result-viewer">
    <div class="images-comparison">
      <div class="image-box" v-if="inputImage">
        <span class="badge">原图 (Before)</span>
        <img :src="inputImage" alt="Original" class="result-img" />
      </div>
      <div class="image-box" v-if="outputImage">
        <span class="badge highlight">重绘图 (After)</span>
        <img :src="outputImage" alt="Output" class="result-img" />
      </div>
    </div>

    <ExecutionSummary :job-detail="jobDetail" />
  </div>
</template>

<style scoped>
.result-viewer {
  display: flex;
  flex-direction: column;
  gap: 32px;
  width: 100%;
}

.images-comparison {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
}

@media (max-width: 768px) {
  .images-comparison {
    grid-template-columns: 1fr;
  }
}

.image-box {
  position: relative;
  background: var(--bg-secondary);
  border-radius: var(--radius-md);
  overflow: hidden;
  border: 1px solid var(--border-glass);
}

.badge {
  position: absolute;
  top: 16px;
  left: 16px;
  padding: 4px 12px;
  background: rgba(0, 0, 0, 0.6);
  backdrop-filter: blur(8px);
  color: white;
  border-radius: 999px;
  font-size: 0.8rem;
  z-index: 10;
}

.badge.highlight {
  background: rgba(99, 102, 241, 0.8);
}

.result-img {
  width: 100%;
  height: 100%;
  object-fit: contain;
  display: block;
}
</style>
