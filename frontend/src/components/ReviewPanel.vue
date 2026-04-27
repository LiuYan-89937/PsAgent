<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import type { JobDetailResponse, SearchEffort } from '@/types/api'

const props = defineProps<{
  jobId: string
  payload?: any
  message?: string
  jobDetail?: JobDetailResponse | null
  searchEffort: SearchEffort
}>()

const emit = defineEmits<{
  (e: 'resume', approved: boolean, note: string, searchEffort: SearchEffort): void
  (e: 'update:searchEffort', value: SearchEffort): void
}>()

const note = ref('')
const selectedEffort = ref<SearchEffort>(props.searchEffort)
const inputImage = computed(() => props.jobDetail?.input_assets?.[0]?.content_url || '')
const outputImage = computed(() => props.jobDetail?.selected_output?.content_url || '')
const effortOptions: { value: SearchEffort; label: string; range: string }[] = [
  { value: 'standard', label: '标准', range: '4-6' },
  { value: 'high', label: '高', range: '6-8' },
  { value: 'ultra', label: '超高', range: '8-12' },
]

watch(() => props.searchEffort, (value) => {
  selectedEffort.value = value
})

function selectEffort(value: SearchEffort) {
  selectedEffort.value = value
  emit('update:searchEffort', value)
}

function handleApprove() {
  emit('resume', true, note.value, selectedEffort.value)
}

function handleReject() {
  emit('resume', false, note.value, selectedEffort.value)
}
</script>

<template>
  <div class="review-panel glass-panel">
    <div class="warning-icon">
      <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" class="stroke-warning" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
        <line x1="12" y1="9" x2="12" y2="13"/>
        <line x1="12" y1="17" x2="12.01" y2="17"/>
      </svg>
    </div>
    
    <h2>需要人工确认</h2>
    <p class="review-message">{{ message || '模型在修改过程中遇到了不确定因素，需要您的审核。' }}</p>

    <div class="preview-grid" v-if="inputImage || outputImage">
      <div class="preview-card" v-if="inputImage">
        <span class="preview-badge">原图</span>
        <img :src="inputImage" alt="原图" class="preview-image" />
      </div>
      <div class="preview-card" v-if="outputImage">
        <span class="preview-badge highlight">当前效果</span>
        <img :src="outputImage" alt="当前效果" class="preview-image" />
      </div>
    </div>

    <div v-if="payload" class="payload-box">
      <div v-if="payload.reason" class="payload-item">
        <span class="label">原因</span>
        <span class="value">{{ payload.reason }}</span>
      </div>
      <div v-if="payload.summary" class="payload-item">
        <span class="label">分析摘要</span>
        <span class="value">{{ payload.summary }}</span>
      </div>
      <div v-if="payload.suggested_action" class="payload-item">
        <span class="label">建议操作</span>
        <span class="value action">{{ payload.suggested_action }}</span>
      </div>
    </div>

    <div class="input-section">
      <textarea 
        v-model="note" 
        class="input-base" 
        placeholder="看过当前效果后，可以在这里输入下一步调整提示，例如：主体再亮一点，背景压暗一些。"
        rows="3"
      ></textarea>
    </div>

    <div class="effort-row">
      <span class="effort-label">搜索强度</span>
      <div class="effort-control">
        <button
          v-for="option in effortOptions"
          :key="option.value"
          type="button"
          class="effort-option"
          :class="{ active: option.value === selectedEffort }"
          @click="selectEffort(option.value)"
        >
          <span>{{ option.label }}</span>
          <small>{{ option.range }}轮</small>
        </button>
      </div>
    </div>

    <div class="actions">
      <button class="btn-secondary decline-btn" @click="handleReject">拒绝并中止</button>
      <button class="btn-primary" @click="handleApprove">按当前意见继续</button>
    </div>
  </div>
</template>

<style scoped>
.review-panel {
  padding: 32px;
  width: 100%;
  border-color: rgba(245, 158, 11, 0.4);
  box-shadow: 0 4px 32px rgba(245, 158, 11, 0.15);
  animation: slideIn 0.4s ease-out;
}

@keyframes slideIn {
  from { opacity: 0; transform: translateY(20px) scale(0.98); }
  to { opacity: 1; transform: translateY(0) scale(1); }
}

.warning-icon {
  margin-bottom: 16px;
  display: inline-flex;
  padding: 12px;
  border-radius: 50%;
  background: rgba(245, 158, 11, 0.1);
}

.stroke-warning {
  stroke: var(--status-warning);
}

h2 {
  margin: 0 0 8px 0;
  font-size: 1.5rem;
  color: var(--status-warning);
}

.review-message {
  margin: 0 0 24px 0;
  color: var(--text-main);
  font-size: 1rem;
}

.preview-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px;
  margin-bottom: 24px;
}

.preview-card {
  position: relative;
  background: rgba(0, 0, 0, 0.18);
  border: 1px solid var(--border-glass);
  border-radius: var(--radius-sm);
  overflow: hidden;
  min-height: 180px;
}

.preview-image {
  display: block;
  width: 100%;
  height: 100%;
  object-fit: contain;
  background: rgba(0, 0, 0, 0.16);
}

.preview-badge {
  position: absolute;
  top: 12px;
  left: 12px;
  z-index: 1;
  padding: 4px 10px;
  border-radius: 999px;
  background: rgba(0, 0, 0, 0.58);
  color: #fff;
  font-size: 0.78rem;
}

.preview-badge.highlight {
  background: rgba(99, 102, 241, 0.85);
}

.effort-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  margin: 16px 0 24px;
}

.effort-label {
  color: var(--text-muted);
  font-size: 0.9rem;
}

.effort-control {
  display: inline-grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 4px;
  padding: 4px;
  border-radius: var(--radius-sm);
  background: rgba(255, 255, 255, 0.06);
  border: 1px solid var(--border-glass);
}

.effort-option {
  min-width: 84px;
  border: 0;
  border-radius: 6px;
  padding: 7px 10px;
  color: var(--text-muted);
  background: transparent;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
}

.effort-option.active {
  color: var(--text-inverse);
  background: rgba(255, 255, 255, 0.14);
}

.effort-option small {
  font-size: 0.72rem;
  color: inherit;
  opacity: 0.72;
}

.payload-box {
  background: rgba(0, 0, 0, 0.2);
  border-radius: var(--radius-sm);
  padding: 16px;
  margin-bottom: 24px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.payload-item {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.label {
  font-size: 0.8rem;
  color: var(--text-muted);
  text-transform: uppercase;
}

.value {
  font-size: 0.95rem;
  color: var(--text-main);
}

.value.action {
  color: var(--accent-primary);
}

.input-section {
  margin-bottom: 24px;
}

.input-section textarea {
  width: 100%;
  resize: vertical;
}

.actions {
  display: flex;
  gap: 16px;
  justify-content: flex-end;
}

@media (max-width: 768px) {
  .preview-grid {
    grid-template-columns: 1fr;
  }
}

.decline-btn:hover {
  background: rgba(239, 68, 68, 0.1);
  color: var(--status-error);
  border-color: rgba(239, 68, 68, 0.3);
}
</style>
