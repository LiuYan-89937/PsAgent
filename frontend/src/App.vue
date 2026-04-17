<script setup lang="ts">
import { ref, onBeforeUnmount } from 'vue'
import { streamEdit, getJob, resumeReview } from '@/lib/api'
import type { AssetResponse, SseEventPayload, JobDetailResponse } from '@/types/api'

import ImageUploader from '@/components/ImageUploader.vue'
import PromptInput from '@/components/PromptInput.vue'
import ProcessViewer from '@/components/ProcessViewer.vue'
import ReviewPanel from '@/components/ReviewPanel.vue'
import ResultViewer from '@/components/ResultViewer.vue'

type AppState = 'idle' | 'ready' | 'processing' | 'review_required' | 'completed' | 'fatal_error'

const currentState = ref<AppState>('idle')
const currentAsset = ref<AssetResponse | null>(null)
const currentJobId = ref<string | null>(null)
const sseEvents = ref<SseEventPayload[]>([])

const reviewPayload = ref<any>(null)
const reviewMessage = ref<string>('')
const jobDetail = ref<JobDetailResponse | null>(null)
const errorMsg = ref('')
const errorDetail = ref<Record<string, unknown> | null>(null)
const pollTimer = ref<number | null>(null)

function clearPollTimer() {
  if (pollTimer.value != null) {
    window.clearInterval(pollTimer.value)
    pollTimer.value = null
  }
}

function extractLatestInterrupt(detail: JobDetailResponse): { payload: any; message: string } | null {
  const latestInterrupt = [...detail.events]
    .reverse()
    .find((event) => event.event === 'interrupt')

  if (!latestInterrupt) return null
  return {
    payload: latestInterrupt.payload ?? null,
    message: typeof latestInterrupt.message === 'string' ? latestInterrupt.message : (detail.job.current_message || ''),
  }
}

async function fetchJobDetailForReview() {
  if (!currentJobId.value) return
  try {
    const detail = await getJob(currentJobId.value)
    jobDetail.value = detail
  } catch (err) {
    console.warn('获取审核详情失败', err)
  }
}

function handleUploadSuccess(asset: AssetResponse) {
  currentAsset.value = asset
  currentState.value = 'ready'
}

function handleUploadError(err: string) {
  errorMsg.value = err
  errorDetail.value = null
  setTimeout(() => { errorMsg.value = '' }, 3000)
}

function handleCancelUpload() {
  currentAsset.value = null
  currentState.value = 'idle'
}

async function handleStartEdit(instruction: string) {
  if (!currentAsset.value) return

  clearPollTimer()
  currentState.value = 'processing'
  sseEvents.value = []
  currentJobId.value = null
  reviewPayload.value = null
  reviewMessage.value = ''
  jobDetail.value = null
  errorMsg.value = ''
  errorDetail.value = null
  
  const payload = {
    user_id: 'web-user',
    instruction,
    input_asset_ids: [currentAsset.value.asset_id]
  }

  try {
    await streamEdit(payload, (eventName, data) => {
      // 记录事件
      sseEvents.value.push(data)

      if (eventName === 'job_created' && data.job_id) {
        currentJobId.value = data.job_id
      }
      
      if (eventName === 'interrupt') {
        reviewPayload.value = data.payload
        reviewMessage.value = data.message || ''
        currentState.value = 'review_required'
        void fetchJobDetailForReview()
      }

      if (eventName === 'job_interrupted') {
        // 如果后端先发 interrupt 再发 job_interrupted，上面已经捕获了
        if (currentState.value !== 'review_required') {
          currentState.value = 'review_required'
        }
        void fetchJobDetailForReview()
      }

      if (eventName === 'job_completed') {
        fetchJobDetailAndComplete()
      }

      if (eventName === 'job_failed') {
        errorMsg.value = '处理失败: ' + (data.message || '未知错误')
        errorDetail.value = (data.error_detail as Record<string, unknown> | undefined) || null
        currentState.value = 'fatal_error'
      }
    })
  } catch (err) {
    console.error('Stream Error:', err)
    // 可能是打断引起的流断开（目前设计流是一次性的），如果有 job_id 且进入了 review_required 状态则忽略报错
    if (currentState.value !== 'review_required' && currentState.value !== 'completed') {
      errorMsg.value = err instanceof Error ? err.message : '连接失败'
      errorDetail.value = err instanceof Error ? { type: err.name, message: err.message } : null
      currentState.value = 'fatal_error'
    }
  }
}

async function fetchJobDetailAndComplete() {
  if (!currentJobId.value) return
  try {
    clearPollTimer()
    const detail = await getJob(currentJobId.value)
    jobDetail.value = detail
    currentState.value = 'completed'
  } catch (err) {
    errorMsg.value = '获取结果详情失败'
    errorDetail.value = err instanceof Error ? { type: err.name, message: err.message } : null
    currentState.value = 'fatal_error'
  }
}

async function handleResumeReview(approved: boolean, note: string) {
  if (!currentJobId.value) return

  clearPollTimer()
  // 恢复状态到 processing 显示
  currentState.value = 'processing'
  try {
    await resumeReview({
      job_id: currentJobId.value,
      approved,
      note
    })
    // 恢复执行后，轮询或等待结果，因为当前没有重建 SSE。
    // 在文档中提到：前端仍建议在恢复后重新拉一次任务详情。
    // 为了体验，我们可以轮询 getJob 直到 completed/failed
    pollJobStatus(currentJobId.value)
  } catch (err) {
    errorMsg.value = '从人工审核恢复失败'
    errorDetail.value = err instanceof Error ? { type: err.name, message: err.message } : null
    currentState.value = 'fatal_error'
  }
}

async function pollJobStatus(jobId: string) {
  clearPollTimer()
  pollTimer.value = window.setInterval(async () => {
    try {
      const detail = await getJob(jobId)
      if (detail.job.status === 'completed') {
        clearPollTimer()
        jobDetail.value = detail
        currentState.value = 'completed'
      } else if (detail.job.status === 'failed') {
        clearPollTimer()
        errorMsg.value = '任务最终执行失败'
        errorDetail.value = detail.job.error_detail || null
        currentState.value = 'fatal_error'
      } else if (detail.job.status === 'review_required') {
        clearPollTimer()
        const interrupt = extractLatestInterrupt(detail)
        reviewPayload.value = interrupt?.payload ?? null
        reviewMessage.value = interrupt?.message || detail.job.current_message || '等待人工确认'
        jobDetail.value = detail
        currentState.value = 'review_required'
      }
    } catch {
      // 忽略单次网络错误
    }
  }, 2000)
}

function handleReset() {
  clearPollTimer()
  currentState.value = 'idle'
  currentAsset.value = null
  currentJobId.value = null
  sseEvents.value = []
  jobDetail.value = null
  reviewPayload.value = null
  reviewMessage.value = ''
  errorMsg.value = ''
  errorDetail.value = null
}

onBeforeUnmount(() => {
  clearPollTimer()
})
</script>

<template>
  <main class="app-layout">
    <header class="app-header">
      <div class="logo">
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="logo-icon"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>
        <span class="font-semibold">PsAgent</span>
      </div>
      <div v-if="errorMsg" class="error-toast">
        {{ errorMsg }}
      </div>
    </header>

    <div class="app-body" :class="{ 'app-body-top': currentState === 'completed' }">
      <transition name="slide-up" mode="out-in">
        
        <!-- State: IDLE -->
        <div class="view-container" v-if="currentState === 'idle'" key="idle">
          <div class="hero">
            <h1>让 AI 为你修图</h1>
            <p>上传任意图片，用自然语言告诉 Agent 你想怎么修改。支持区域重绘、画面控制和精准调参。</p>
          </div>
          <ImageUploader 
            @upload-success="handleUploadSuccess"
            @upload-error="handleUploadError"
          />
        </div>

        <!-- State: READY -->
        <div class="view-container" v-else-if="currentState === 'ready' && currentAsset" key="ready">
          <PromptInput 
            :asset="currentAsset"
            @submit="handleStartEdit"
            @cancel="handleCancelUpload"
          />
        </div>

        <!-- State: PROCESSING -->
        <div class="view-container" v-else-if="currentState === 'processing'" key="processing">
          <ProcessViewer :events="sseEvents" />
        </div>

        <!-- State: REVIEW_REQUIRED -->
        <div class="view-container" v-else-if="currentState === 'review_required'" key="review">
          <ReviewPanel 
            :job-id="currentJobId!"
            :payload="reviewPayload"
            :message="reviewMessage"
            :job-detail="jobDetail"
            @resume="handleResumeReview"
          />
        </div>

        <!-- State: COMPLETED -->
        <div class="view-container view-wide" v-else-if="currentState === 'completed' && jobDetail" key="completed">
          <div class="completed-header">
            <h2>处理完成</h2>
            <button class="btn-secondary" @click="handleReset">处理新图片</button>
          </div>
          <ResultViewer :job-detail="jobDetail" />
        </div>

        <!-- State: FATAL ERROR -->
        <div class="view-container" v-else-if="currentState === 'fatal_error'" key="error">
          <div class="glass-panel text-center" style="padding:48px;">
            <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" class="stroke-error mx-auto mb-4" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <circle cx="12" cy="12" r="10"></circle>
              <line x1="12" y1="8" x2="12" y2="12"></line>
              <line x1="12" y1="16" x2="12.01" y2="16"></line>
            </svg>
            <h2>系统似乎遇到了一些问题</h2>
            <p style="color:var(--text-muted); margin-bottom:24px;">{{ errorMsg }}</p>
            <div v-if="errorDetail" class="error-detail-card">
              <div v-if="errorDetail.stage" class="error-detail-row">
                <span class="error-detail-label">阶段</span>
                <span>{{ errorDetail.stage }}</span>
              </div>
              <div v-if="errorDetail.node" class="error-detail-row">
                <span class="error-detail-label">节点</span>
                <span>{{ errorDetail.node }}</span>
              </div>
              <div v-if="errorDetail.op" class="error-detail-row">
                <span class="error-detail-label">工具包</span>
                <span>{{ errorDetail.op }}</span>
              </div>
              <div v-if="errorDetail.region" class="error-detail-row">
                <span class="error-detail-label">区域</span>
                <span>{{ errorDetail.region }}</span>
              </div>
              <div v-if="errorDetail.type" class="error-detail-row">
                <span class="error-detail-label">异常类型</span>
                <span>{{ errorDetail.type }}</span>
              </div>
              <div v-if="errorDetail.message" class="error-detail-row">
                <span class="error-detail-label">异常信息</span>
                <span>{{ errorDetail.message }}</span>
              </div>
              <details v-if="errorDetail.traceback" class="error-detail-trace">
                <summary>查看详细堆栈</summary>
                <pre>{{ errorDetail.traceback }}</pre>
              </details>
            </div>
            <button class="btn-primary" @click="handleReset">返回首页重试</button>
          </div>
        </div>

      </transition>
    </div>
  </main>
</template>

<style scoped>
.app-layout {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.app-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 32px;
  background: rgba(10, 10, 11, 0.5);
  backdrop-filter: blur(8px);
  border-bottom: 1px solid var(--border-glass);
}

.logo {
  display: flex;
  align-items: center;
  gap: 12px;
  color: var(--text-inverse);
  font-size: 1.1rem;
  letter-spacing: 0.5px;
}

.logo-icon {
  color: var(--accent-primary);
}

.error-detail-card {
  width: min(100%, 720px);
  margin: 0 auto 24px;
  padding: 16px;
  border-radius: 16px;
  background: rgba(127, 29, 29, 0.18);
  border: 1px solid rgba(248, 113, 113, 0.25);
  text-align: left;
}

.error-detail-row {
  display: grid;
  grid-template-columns: 92px 1fr;
  gap: 12px;
  margin-bottom: 10px;
  font-size: 0.92rem;
  color: var(--text-main);
}

.error-detail-label {
  color: #fca5a5;
}

.error-detail-trace {
  margin-top: 12px;
}

.error-detail-trace summary {
  cursor: pointer;
  color: #fecaca;
  margin-bottom: 8px;
}

.error-detail-trace pre {
  margin: 0;
  padding: 12px;
  border-radius: 12px;
  background: rgba(0, 0, 0, 0.28);
  color: #f5f5f5;
  font-size: 0.78rem;
  line-height: 1.5;
  white-space: pre-wrap;
  word-break: break-word;
}

.font-semibold {
  font-weight: 600;
}

.error-toast {
  background: rgba(239, 68, 68, 0.15);
  border: 1px solid rgba(239, 68, 68, 0.3);
  color: #fca5a5;
  padding: 6px 16px;
  border-radius: 999px;
  font-size: 0.85rem;
  animation: slideIn 0.3s ease;
}

.app-body {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 40px 20px;
}

.view-container {
  width: 100%;
  max-width: 680px;
  margin: 0 auto;
}

.view-wide {
  max-width: 1200px;
}

.hero {
  text-align: center;
  margin-bottom: 40px;
}

.hero h1 {
  font-size: 2.5rem;
  margin: 0 0 16px 0;
  background: linear-gradient(135deg, #fff 0%, #a5a5b0 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.hero p {
  color: var(--text-muted);
  font-size: 1.1rem;
  margin: 0 auto;
  max-width: 480px;
}

.completed-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
}

.completed-header h2 {
  margin: 0;
  font-size: 1.5rem;
}

.app-body-top {
  align-items: flex-start;
}

.text-center {
  text-align: center;
}

.mx-auto {
  margin-left: auto;
  margin-right: auto;
}

.mb-4 {
  margin-bottom: 16px;
}

.stroke-error {
  stroke: var(--status-error);
}
</style>
