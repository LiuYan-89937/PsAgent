<script setup lang="ts">
import { ref } from 'vue'
import { uploadAssets } from '@/lib/api'
import type { AssetResponse } from '@/types/api'

const emit = defineEmits<{
  (e: 'upload-success', asset: AssetResponse): void
  (e: 'upload-error', error: string): void
}>()

const isDragging = ref(false)
const isUploading = ref(false)
const fileInput = ref<HTMLInputElement | null>(null)

function onDragOver(e: DragEvent) {
  e.preventDefault()
  isDragging.value = true
}

function onDragLeave(e: DragEvent) {
  e.preventDefault()
  isDragging.value = false
}

async function handleFiles(files: FileList | null) {
  if (!files || files.length === 0) return
  
  // 暂时只取第一张图
  const file = files[0]
  if (!file.type.startsWith('image/')) {
    emit('upload-error', '请上传图片文件')
    return
  }

  isUploading.value = true
  try {
    const res = await uploadAssets([file])
    if (res.items && res.items.length > 0) {
      emit('upload-success', res.items[0])
    } else {
      emit('upload-error', '上传失败：无返回图片')
    }
  } catch (error) {
    emit('upload-error', error instanceof Error ? error.message : '上传失败')
  } finally {
    isUploading.value = false
    isDragging.value = false
    if (fileInput.value) fileInput.value.value = ''
  }
}

function onDrop(e: DragEvent) {
  e.preventDefault()
  handleFiles(e.dataTransfer?.files || null)
}

function onFileSelect(e: Event) {
  const target = e.target as HTMLInputElement
  handleFiles(target.files)
}
</script>

<template>
  <div 
    class="upload-zone glass-panel" 
    :class="{ 'is-dragging': isDragging, 'is-uploading': isUploading }"
    @dragover="onDragOver"
    @dragleave="onDragLeave"
    @drop="onDrop"
    @click="() => fileInput?.click()"
  >
    <div class="upload-content">
      <div class="icon-wrapper">
        <svg v-if="!isUploading" xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
          <polyline points="17 8 12 3 7 8"/>
          <line x1="12" y1="3" x2="12" y2="15"/>
        </svg>
        <svg v-else class="spinner" xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
          <line x1="12" y1="2" x2="12" y2="6"/>
          <line x1="12" y1="18" x2="12" y2="22"/>
          <line x1="4.93" y1="4.93" x2="7.76" y2="7.76"/>
          <line x1="16.24" y1="16.24" x2="19.07" y2="19.07"/>
          <line x1="2" y1="12" x2="6" y2="12"/>
          <line x1="18" y1="12" x2="22" y2="12"/>
          <line x1="4.93" y1="19.07" x2="7.76" y2="16.24"/>
          <line x1="16.24" y1="7.76" x2="19.07" y2="4.93"/>
        </svg>
      </div>
      
      <h3>{{ isUploading ? '正在上传图片...' : '上传主图片' }}</h3>
      <p class="desc">{{ isUploading ? '稍等片刻，资源即将就绪' : '拖拽图片至此处，或点击选择图片' }}</p>
    </div>

    <input 
      type="file" 
      ref="fileInput" 
      class="hidden-input" 
      accept="image/*"
      @change="onFileSelect"
    />
  </div>
</template>

<style scoped>
.upload-zone {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  width: 100%;
  min-height: 320px;
  border: 2px dashed var(--border-glass);
  cursor: pointer;
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
}

.upload-zone:hover, .upload-zone.is-dragging {
  border-color: var(--accent-primary);
  background: rgba(99, 102, 241, 0.05);
  transform: scale(1.02);
}

.upload-zone.is-uploading {
  pointer-events: none;
  opacity: 0.8;
}

.upload-content {
  text-align: center;
  padding: 32px;
}

.icon-wrapper {
  margin-bottom: 20px;
  color: var(--text-muted);
  transition: color 0.3s ease;
}

.upload-zone:hover .icon-wrapper {
  color: var(--accent-primary);
}

h3 {
  margin: 0 0 8px 0;
  font-size: 1.25rem;
  font-weight: 500;
  color: var(--text-main);
}

.desc {
  margin: 0;
  font-size: 0.95rem;
  color: var(--text-muted);
}

.hidden-input {
  display: none;
}

.spinner {
  animation: spin 1.5s linear infinite;
  color: var(--accent-primary);
}

@keyframes spin {
  100% { transform: rotate(360deg); }
}
</style>
