<script setup lang="ts">
import { ref } from 'vue'
import type { AssetResponse } from '@/types/api'

const props = defineProps<{
  asset: AssetResponse
}>()

const emit = defineEmits<{
  (e: 'submit', instruction: string): void
  (e: 'cancel'): void
}>()

const instruction = ref('')
const isFocus = ref(false)

function handleSubmit() {
  if (!instruction.value.trim()) return
  emit('submit', instruction.value.trim())
}
</script>

<template>
  <div class="prompt-container glass-panel">
    <div class="preview-header">
      <div class="preview-img-wrapper">
        <img :src="asset.content_url" alt="Preview" class="preview-img" />
      </div>
      <div class="asset-info">
        <h3>当前图像已就绪</h3>
        <p>{{ asset.filename }}</p>
      </div>
      <button class="btn-cancel" @click="$emit('cancel')" title="重新上传">
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <line x1="18" y1="6" x2="6" y2="18"></line>
          <line x1="6" y1="6" x2="18" y2="18"></line>
        </svg>
      </button>
    </div>

    <div class="input-area" :class="{ 'is-focus': isFocus }">
      <textarea 
        v-model="instruction"
        placeholder="告诉 Agent 你想怎么修改这张图片？例如：将背景替换为星空"
        class="prompt-textarea input-base"
        rows="3"
        @focus="isFocus = true"
        @blur="isFocus = false"
        @keydown.enter.prevent="handleSubmit"
      ></textarea>
      <div class="action-bar">
        <span class="hint">按 Enter 键快速提交指令</span>
        <button 
          class="btn-primary" 
          :disabled="!instruction.trim()" 
          @click="handleSubmit"
        >
          开始处理
          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-left:8px;">
            <line x1="22" y1="2" x2="11" y2="13"></line>
            <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
          </svg>
        </button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.prompt-container {
  padding: 24px;
  width: 100%;
}

.preview-header {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 24px;
  padding-bottom: 24px;
  border-bottom: 1px solid var(--border-glass);
}

.preview-img-wrapper {
  width: 72px;
  height: 72px;
  border-radius: var(--radius-sm);
  overflow: hidden;
  box-shadow: 0 4px 12px rgba(0,0,0,0.2);
  border: 1px solid var(--border-glass);
}

.preview-img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.asset-info {
  flex: 1;
}

.asset-info h3 {
  margin: 0 0 4px 0;
  font-size: 1.1rem;
  font-weight: 500;
  color: var(--text-inverse);
}

.asset-info p {
  margin: 0;
  font-size: 0.85rem;
  color: var(--text-muted);
}

.btn-cancel {
  background: transparent;
  border: none;
  color: var(--text-muted);
  cursor: pointer;
  padding: 8px;
  border-radius: 50%;
  transition: all 0.2s;
}

.btn-cancel:hover {
  color: var(--status-error);
  background: rgba(239, 68, 68, 0.1);
}

.input-area {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.prompt-textarea {
  width: 100%;
  resize: none;
  font-size: 1rem;
  line-height: 1.5;
  background: rgba(0,0,0,0.3);
}

.action-bar {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.hint {
  font-size: 0.8rem;
  color: var(--text-muted);
}
</style>
