<script setup lang="ts">
import { computed, ref } from 'vue'
import type { JobDetailResponse } from '@/types/api'

const props = defineProps<{
  jobId: string
  payload?: any
  message?: string
  jobDetail?: JobDetailResponse | null
}>()

const emit = defineEmits<{
  (e: 'resume', approved: boolean, note: string): void
}>()

const note = ref('')
const inputImage = computed(() => props.jobDetail?.input_assets?.[0]?.content_url || '')
const outputImage = computed(() => props.jobDetail?.selected_output?.content_url || '')

function handleApprove() {
  emit('resume', true, note.value)
}

function handleReject() {
  emit('resume', false, note.value)
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
