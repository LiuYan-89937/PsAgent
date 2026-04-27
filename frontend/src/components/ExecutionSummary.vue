<script setup lang="ts">
import { computed } from 'vue'
import type {
  CandidatePreviewExecution,
  EvaluationReport,
  ExecutionTraceItem,
  FallbackTraceItem,
  FocusKey,
  JobDetailResponse,
  JobEvent,
  RoundAction,
  SearchCandidateResponse,
  SearchRoundResponse,
} from '@/types/api'

const props = defineProps<{
  jobDetail: JobDetailResponse
  showTrace: boolean
  showDebug: boolean
}>()

type StatusTone = 'success' | 'warning' | 'error' | 'neutral'

const FOCUS_LABELS: Record<FocusKey, string> = {
  global_tone: '全局影调',
  subject_separation: '主体分离',
  subject_cleanup: '主体清理',
  finish: '最终收尾',
}

const ACTION_LABELS: Record<RoundAction, string> = {
  keep: '保留',
  recover_same_round: '同轮恢复',
  stop_round: '停止本轮',
}

function focusLabel(focus?: FocusKey | string | null): string {
  if (!focus) return '未标记焦点'
  return FOCUS_LABELS[focus as FocusKey] ?? String(focus)
}

function actionLabel(action?: RoundAction | string | null): string {
  if (!action) return '未给出'
  return ACTION_LABELS[action as RoundAction] ?? String(action)
}

function imageUrl(execution?: CandidatePreviewExecution | null): string | null {
  return execution?.output_asset?.content_url || null
}

function outputFilename(execution?: CandidatePreviewExecution | null): string | null {
  return execution?.output_asset?.filename || execution?.output_asset_id || execution?.output_image_path || null
}

function percent(score?: number | null): string {
  if (typeof score !== 'number' || Number.isNaN(score)) return '未评分'
  const normalized = score <= 1 ? score * 100 : score
  return `${Math.round(normalized)}`
}

function traceSummary(trace?: ExecutionTraceItem[] | null): string {
  const items = trace || []
  const ok = items.filter((item) => item.ok !== false).length
  const failed = items.filter((item) => item.ok === false).length
  const fallback = items.filter((item) => item.fallback_used).length
  return `${items.length} 步 / ${ok} 成功 / ${failed} 失败 / ${fallback} 回退`
}

function candidateTone(candidate: SearchCandidateResponse): StatusTone {
  if (candidate.selected) return 'success'
  if (candidate.review?.recommended_action === 'recover_same_round') return 'warning'
  if (candidate.review?.overall_ok === false) return 'error'
  return 'neutral'
}

function eventTitle(event: JobEvent): string {
  const label = event.event.replaceAll('_', ' ')
  if (event.round) return `${label} · ${event.round}`
  if (event.node) return `${label} · ${event.node}`
  return label
}

function fallbackMeta(item: FallbackTraceItem): string {
  return [item.round_id, item.focus ? focusLabel(item.focus) : null, item.candidate_id, item.source, item.location]
    .filter(Boolean)
    .join(' / ')
}

const objectiveLines = computed(() => {
  const card = props.jobDetail.objective_card
  if (!card) return []
  return [
    ...card.goals.map((goal) => ({ label: '目标', text: `${goal.kind} · ${goal.target_region}` })),
    ...card.gaps.map((gap) => ({ label: '缺口', text: gap.description })),
    ...card.preserve.map((text) => ({ label: '保留', text })),
    ...card.constraints.map((text) => ({ label: '约束', text })),
  ].slice(0, 10)
})

const rounds = computed<SearchRoundResponse[]>(() => props.jobDetail.rounds || [])

const finalReview = computed<EvaluationReport | null>(() => props.jobDetail.final_review || props.jobDetail.eval_report || null)

const finalTrace = computed<ExecutionTraceItem[]>(() => (
  props.jobDetail.final_execution_trace?.length
    ? props.jobDetail.final_execution_trace
    : props.jobDetail.execution_trace || []
))

const fallbackItems = computed<FallbackTraceItem[]>(() => props.jobDetail.fallback_trace || [])

const timelineEvents = computed<JobEvent[]>(() => (
  (props.jobDetail.events || []).filter((event) => event.message).slice(-24)
))
</script>

<template>
  <div class="execution-summary">
    <section v-if="jobDetail.objective_card" class="glass-panel summary-block">
      <div class="section-head">
        <div>
          <p class="eyebrow">Objective</p>
          <h3>{{ jobDetail.objective_card.summary || '自动修图目标' }}</h3>
        </div>
        <span class="pill">{{ jobDetail.objective_card.mode }}</span>
      </div>
      <div v-if="objectiveLines.length" class="objective-grid">
        <div v-for="item in objectiveLines" :key="`${item.label}-${item.text}`" class="objective-item">
          <span>{{ item.label }}</span>
          <p>{{ item.text }}</p>
        </div>
      </div>
    </section>

    <section class="glass-panel summary-block">
      <div class="section-head">
        <div>
          <p class="eyebrow">Rounds</p>
          <h3>搜索轮次</h3>
        </div>
        <span class="pill">{{ rounds.length }} 轮</span>
      </div>

      <div v-if="rounds.length" class="round-list">
        <article v-for="round in rounds" :key="round.id" class="round-card">
          <div class="round-header">
            <div>
              <p class="eyebrow">Round {{ round.index }}</p>
              <h4>{{ focusLabel(round.focus) }}</h4>
            </div>
            <span class="pill" :class="{ success: round.completed }">
              {{ round.completed ? '已完成' : '未完成' }}
            </span>
          </div>

          <div v-if="round.objective_gaps.length" class="gap-list">
            <div v-for="gap in round.objective_gaps" :key="gap.id" class="gap-item">
              <span>P{{ gap.priority }}</span>
              <p>{{ gap.description }}</p>
            </div>
          </div>

          <div v-if="round.guidance" class="guidance-card">
            <div>
              <p class="eyebrow">Round guidance</p>
              <h5>{{ round.guidance.target_prompt || '本轮导向提示词' }}</h5>
              <p v-if="round.guidance.visual_diagnosis">{{ round.guidance.visual_diagnosis }}</p>
            </div>
            <div v-if="round.guidance.preserve.length || round.guidance.avoid.length" class="guidance-tags">
              <span v-for="item in round.guidance.preserve" :key="`preserve-${round.id}-${item}`">保留 · {{ item }}</span>
              <span v-for="item in round.guidance.avoid" :key="`avoid-${round.id}-${item}`">避免 · {{ item }}</span>
            </div>
          </div>

          <div class="candidate-grid">
            <article
              v-for="candidate in round.candidates"
              :key="candidate.candidate_id"
              class="candidate-card"
              :class="[candidateTone(candidate), { selected: candidate.selected }]"
            >
              <div class="candidate-top">
                <div>
                  <h5>{{ candidate.label || candidate.candidate_id }}</h5>
                  <p>{{ candidate.program?.summary || '无额外说明' }}</p>
                </div>
                <span class="score">{{ percent(candidate.review?.score) }}</span>
              </div>
              <img
                v-if="imageUrl(candidate.preview_execution)"
                :src="imageUrl(candidate.preview_execution)!"
                :alt="candidate.label || candidate.candidate_id"
                class="preview-image"
              />
              <div class="candidate-meta">
                <span>{{ candidate.program?.steps?.length ?? 0 }} 步</span>
                <span>{{ traceSummary(candidate.preview_execution?.execution_trace) }}</span>
                <span>{{ actionLabel(candidate.review?.recommended_action) }}</span>
              </div>
              <p v-if="candidate.review?.summary" class="candidate-summary">{{ candidate.review.summary }}</p>
              <p v-if="candidate.eliminated_reason" class="eliminated">淘汰原因：{{ candidate.eliminated_reason }}</p>
            </article>
          </div>

          <div v-if="round.selected_full_execution" class="commit-card">
            <div>
              <p class="eyebrow">Selected full-res commit</p>
              <h5>{{ round.selected_candidate_id || '未记录候选' }}</h5>
              <p>{{ outputFilename(round.selected_full_execution) || '正式输出已生成' }}</p>
            </div>
            <span>{{ traceSummary(round.selected_full_execution.execution_trace) }}</span>
          </div>

          <div v-if="round.round_review" class="review-card">
            <div>
              <p class="eyebrow">Round review</p>
              <h5>{{ actionLabel(round.round_review.recommended_action) }} · {{ percent(round.round_review.score) }}</h5>
            </div>
            <p>{{ round.round_review.summary || '本轮未记录额外说明。' }}</p>
          </div>

          <div v-if="round.recovery_decision?.triggered" class="recovery-card">
            <div>
              <p class="eyebrow">Recovery</p>
              <h5>{{ round.recovery_decision.reason || '触发同轮恢复' }}</h5>
            </div>
            <p>
              {{ round.recovery_decision.candidate_ids.length }} 个恢复候选，
              选中 {{ round.recovery_decision.selected_candidate_id || '未选中' }}
            </p>
          </div>
        </article>
      </div>

      <p v-else class="empty-state">暂无 round artifact。</p>
    </section>

    <section v-if="finalReview" class="glass-panel summary-block">
      <div class="section-head">
        <div>
          <p class="eyebrow">Final review</p>
          <h3>最终评估</h3>
        </div>
        <span class="pill" :class="{ success: finalReview.overall_ok, warning: finalReview.should_request_review }">
          {{ finalReview.should_request_review ? '需复核' : '完成' }}
        </span>
      </div>
      <p class="review-summary">{{ finalReview.summary }}</p>
      <div class="final-stats">
        <span>{{ finalReview.success_count }} 成功</span>
        <span>{{ finalReview.failure_count }} 失败</span>
        <span>{{ finalReview.fallback_count }} 回退</span>
      </div>
    </section>

    <section v-if="showTrace" class="glass-panel summary-block">
      <div class="section-head">
        <div>
          <p class="eyebrow">Trace</p>
          <h3>最终执行链</h3>
        </div>
        <span class="pill">{{ finalTrace.length }} 条</span>
      </div>
      <div v-if="finalTrace.length" class="trace-list">
        <div v-for="(item, index) in finalTrace" :key="`${item.candidate_id || 'trace'}-${index}`" class="trace-row">
          <span class="trace-op">{{ item.op || 'noop' }}</span>
          <span>{{ item.region || 'whole_image' }}</span>
          <span>{{ item.round_id || item.focus || 'final' }}</span>
          <span :class="item.ok === false ? 'status-error' : 'status-success'">
            {{ item.ok === false ? '失败' : '成功' }}
          </span>
          <span v-if="item.fallback_used" class="status-warning">fallback</span>
        </div>
      </div>
      <p v-else class="empty-state">没有最终执行 trace。</p>
    </section>

    <section v-if="showDebug && fallbackItems.length" class="glass-panel summary-block">
      <div class="section-head">
        <div>
          <p class="eyebrow">Fallback</p>
          <h3>回退影响</h3>
        </div>
        <span class="pill warning">{{ fallbackItems.length }} 条</span>
      </div>
      <div class="fallback-list">
        <div v-for="(item, index) in fallbackItems" :key="`${item.source || 'fallback'}-${index}`" class="fallback-row">
          <strong>{{ item.strategy || 'fallback' }}</strong>
          <span>{{ fallbackMeta(item) || '未标记' }}</span>
          <p>{{ item.message || item.error || '无说明' }}</p>
        </div>
      </div>
    </section>

    <section v-if="showDebug && jobDetail.round_timings.length" class="glass-panel summary-block">
      <div class="section-head">
        <div>
          <p class="eyebrow">Timing</p>
          <h3>轮次耗时</h3>
        </div>
      </div>
      <div class="timing-grid">
        <div v-for="item in jobDetail.round_timings" :key="`${item.round}-${item.started_at}`" class="timing-card">
          <strong>{{ item.label || item.round }}</strong>
          <span>{{ focusLabel(item.focus) }}</span>
          <p>{{ item.duration_seconds.toFixed(2) }}s · {{ item.status }}</p>
        </div>
      </div>
    </section>

    <section v-if="showDebug && timelineEvents.length" class="glass-panel summary-block">
      <div class="section-head">
        <div>
          <p class="eyebrow">Events</p>
          <h3>事件流</h3>
        </div>
      </div>
      <div class="event-list">
        <div v-for="(event, index) in timelineEvents" :key="`${event.event}-${index}`" class="event-row">
          <strong>{{ eventTitle(event) }}</strong>
          <p>{{ event.message }}</p>
        </div>
      </div>
    </section>
  </div>
</template>

<style scoped>
.execution-summary {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.summary-block {
  padding: 22px;
}

.section-head,
.round-header,
.candidate-top,
.commit-card,
.review-card,
.recovery-card,
.guidance-card {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
}

.section-head h3,
.round-header h4,
.candidate-top h5,
.commit-card h5,
.review-card h5,
.recovery-card h5,
.guidance-card h5 {
  margin: 0;
}

.eyebrow {
  margin: 0 0 6px;
  color: var(--text-muted);
  font-size: 0.74rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.pill,
.score {
  flex: 0 0 auto;
  padding: 4px 10px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.08);
  border: 1px solid rgba(255, 255, 255, 0.12);
  color: var(--text-main);
  font-size: 0.78rem;
}

.pill.success,
.candidate-card.success .score,
.status-success {
  color: #bbf7d0;
}

.pill.warning,
.candidate-card.warning .score,
.status-warning {
  color: #fde68a;
}

.status-error,
.candidate-card.error .score {
  color: #fecaca;
}

.objective-grid,
.gap-list,
.candidate-grid,
.trace-list,
.fallback-list,
.event-list,
.timing-grid {
  display: grid;
  gap: 12px;
}

.objective-grid {
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  margin-top: 18px;
}

.objective-item,
.gap-item,
.trace-row,
.fallback-row,
.timing-card,
.event-row {
  padding: 12px;
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(255, 255, 255, 0.08);
}

.objective-item span,
.gap-item span {
  color: var(--text-muted);
  font-size: 0.78rem;
}

.objective-item p,
.gap-item p,
.candidate-summary,
.eliminated,
.review-summary,
.fallback-row p,
.event-row p,
.commit-card p,
.review-card p,
.recovery-card p,
.guidance-card p,
.timing-card p {
  margin: 6px 0 0;
  color: var(--text-muted);
  line-height: 1.55;
}

.round-list {
  display: flex;
  flex-direction: column;
  gap: 18px;
  margin-top: 18px;
}

.round-card {
  padding: 16px;
  border-radius: 8px;
  background: rgba(0, 0, 0, 0.16);
  border: 1px solid rgba(255, 255, 255, 0.1);
}

.gap-list {
  margin: 14px 0;
}

.gap-item {
  display: flex;
  align-items: flex-start;
  gap: 10px;
}

.candidate-grid {
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
}

.guidance-card {
  margin: 14px 0;
  padding: 14px;
  border-radius: 8px;
  background: rgba(125, 211, 252, 0.08);
  border: 1px solid rgba(125, 211, 252, 0.18);
}

.guidance-tags {
  display: flex;
  flex: 0 0 220px;
  flex-wrap: wrap;
  gap: 8px;
  justify-content: flex-end;
}

.guidance-tags span {
  padding: 4px 8px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.08);
  color: var(--text-muted);
  font-size: 0.76rem;
}

.candidate-card {
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding: 14px;
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(255, 255, 255, 0.08);
}

.candidate-card.selected {
  border-color: rgba(74, 222, 128, 0.45);
  background: rgba(34, 197, 94, 0.08);
}

.candidate-top p {
  margin: 6px 0 0;
  color: var(--text-muted);
  line-height: 1.45;
}

.preview-image {
  width: 100%;
  max-height: 240px;
  object-fit: contain;
  border-radius: 8px;
  background: rgba(0, 0, 0, 0.18);
}

.candidate-meta,
.final-stats {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  color: var(--text-muted);
  font-size: 0.82rem;
}

.eliminated {
  color: #fecaca;
}

.commit-card,
.review-card,
.recovery-card {
  margin-top: 14px;
  padding: 14px;
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(255, 255, 255, 0.08);
}

.trace-row {
  display: grid;
  grid-template-columns: minmax(120px, 1.3fr) repeat(4, minmax(80px, 1fr));
  gap: 10px;
  align-items: center;
  color: var(--text-muted);
  font-size: 0.88rem;
}

.trace-op {
  color: var(--text-main);
  font-family: 'SF Mono', Monaco, monospace;
}

.fallback-row,
.event-row,
.timing-card {
  color: var(--text-muted);
}

.fallback-row strong,
.event-row strong,
.timing-card strong {
  color: var(--text-main);
}

.empty-state {
  margin: 18px 0 0;
  color: var(--text-muted);
}

@media (max-width: 760px) {
  .section-head,
  .round-header,
  .candidate-top,
  .commit-card,
  .review-card,
  .recovery-card,
  .guidance-card {
    flex-direction: column;
  }

  .guidance-tags {
    flex-basis: auto;
    justify-content: flex-start;
  }

  .trace-row {
    grid-template-columns: 1fr;
  }
}
</style>
