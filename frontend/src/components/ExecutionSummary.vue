<script setup lang="ts">
import { computed } from 'vue'
import type {
  AssetResponse,
  ExecutionTraceItem,
  FallbackTraceItem as ApiFallbackTraceItem,
  JobDetailResponse,
  JobEvent,
  SegmentationTraceItem,
  StageTimingResponse,
} from '@/types/api'

type StatusTone = 'success' | 'warning' | 'error' | 'neutral'

interface TraceGroup {
  key: string
  label: string
  items: TraceViewItem[]
}

interface TraceViewItem {
  sequence: number
  stage?: string | null
  roundKey: string
  roundLabel: string
  op?: string | null
  opLabel: string
  opDescription: string
  region?: string | null
  regionLabel: string
  ok?: boolean | null
  fallbackUsed: boolean
  error?: string | null
  maskPath?: string | null
  paramBadges: string[]
  paramEntries: ParamEntry[]
  outputImageUrl?: string | null
  outputFilename?: string | null
}

interface ParamEntry {
  key: string
  label: string
  value: string
}

interface ToolSummaryItem {
  op: string
  label: string
  description: string
  count: number
  successCount: number
  fallbackCount: number
  failureCount: number
  regions: string[]
  rounds: string[]
}

interface TimelineEntry {
  key: string
  title: string
  message: string
  tone: StatusTone
  meta: string[]
}

interface StageTimingViewItem {
  key: string
  label: string
  durationLabel: string
  tone: StatusTone
  meta: string[]
}

interface FallbackViewItem {
  key: string
  stageLabel: string
  sourceLabel: string
  locationLabel: string
  strategyLabel: string
  message: string
  error?: string | null
}

interface PlannerMaskGroup {
  key: string
  label: string
  items: PlannerMaskItem[]
}

interface PlannerMaskItem {
  sequence: number
  roundKey: string
  roundLabel: string
  op?: string | null
  opLabel: string
  opDescription: string
  region?: string | null
  regionLabel: string
  maskProvider?: string | null
  maskPrompt?: string | null
  maskNegativePrompt?: string | null
  maskSemanticType?: boolean | null
}

interface SegmentationGroup {
  key: string
  label: string
  items: SegmentationViewItem[]
}

interface SegmentationViewItem {
  sequence: number
  roundKey: string
  roundLabel: string
  sourceOp?: string | null
  sourceOpLabel: string
  region?: string | null
  regionLabel: string
  provider?: string | null
  requestedProvider?: string | null
  targetLabel?: string | null
  prompt?: string | null
  negativePrompt?: string | null
  semanticType?: boolean | null
  ok?: boolean | null
  fallbackUsed: boolean
  error?: string | null
  maskPath?: string | null
  apiChain: string[]
  attemptIndex?: number | null
  attemptStrategy?: string | null
  requestedPrompt?: string | null
  effectivePrompt?: string | null
  revertMask?: boolean | null
}

const props = defineProps<{
  jobDetail: JobDetailResponse
  showTrace: boolean
  showDebug: boolean
}>()

const PACKAGE_META: Record<string, { label: string; description: string }> = {
  adjust_exposure: {
    label: '曝光',
    description: '整体或局部提亮、压暗画面。',
  },
  adjust_highlights_shadows: {
    label: '高光阴影',
    description: '拉回高光细节，同时照顾暗部层次。',
  },
  adjust_contrast: {
    label: '对比度',
    description: '提高或降低画面对比关系。',
  },
  adjust_whites_blacks: {
    label: '白场黑场',
    description: '重新设定亮部和暗部的落点。',
  },
  adjust_curves: {
    label: '曲线',
    description: '通过曲线重塑整体明暗结构。',
  },
  adjust_clarity: {
    label: '清晰度',
    description: '增强中间调的局部反差和层次。',
  },
  adjust_texture: {
    label: '纹理',
    description: '提升或柔化中尺度细节纹理。',
  },
  adjust_dehaze: {
    label: '去灰雾',
    description: '压掉雾感，恢复通透度和空气感。',
  },
  adjust_color_mixer: {
    label: '颜色混合',
    description: '按颜色通道分别调整色相、饱和度和明度。',
  },
  adjust_white_balance: {
    label: '白平衡',
    description: '校正色温和偏色，让整体色调更自然。',
  },
  adjust_vibrance_saturation: {
    label: '自然饱和度',
    description: '提高色彩存在感，同时尽量避免过饱和。',
  },
  crop_and_straighten: {
    label: '裁剪拉直',
    description: '调整构图、裁切边缘并修正画面倾斜。',
  },
  denoise: {
    label: '去噪',
    description: '压低亮度或色彩噪点，尽量保住细节。',
  },
  sharpen: {
    label: '锐化',
    description: '增强边缘和细节清晰度。',
  },
}

const REGION_LABELS: Record<string, string> = {
  whole_image: '整张图',
  masked_region: '局部区域',
  main_subject: '主体',
  background: '背景',
  person: '人物',
}

const STAGE_LABELS: Record<string, string> = {
  bootstrap_request: '准备修图请求',
  load_context: '加载上下文',
  analyze_image: '分析图片',
  parse_request: '理解需求',
  plan_execute_round_1: '规划并执行第一轮',
  plan_execute_round_2: '规划并执行第二轮',
  execute_generative: '执行生成式编辑',
  human_review: '人工确认',
  evaluate_result: '评估结果',
  evaluate_round_1: '评估首轮结果',
  evaluate_result_final: '评估最终结果',
  finalize_round_1_result: '确认首轮结果',
  update_memory: '更新记忆',
}

const FALLBACK_SOURCE_LABELS: Record<string, string> = {
  parse_request_model: '需求理解模型',
  analyze_image_model: '图像分析模型',
  planner_model: '规划模型',
  planner_tool_model: '实时规划模型',
  critic_model: '结果评估模型',
  segmentation_provider: '分割服务',
  package_execute: '工具执行',
}

const FALLBACK_STRATEGY_LABELS: Record<string, string> = {
  heuristic_request_intent: '规则归一化',
  basic_image_analysis: '基础图像分析',
  rule_based_plan: '规则规划',
  execution_only_evaluation: '事实评估',
  whole_image_execution: '全图执行',
  skip_local_operation: '跳过局部步骤',
  keep_current_image: '保留当前结果',
  rule_plan_execution: '规则规划并执行',
  finish_current_round: '保留当前轮结果',
  generic_auto_instruction: '通用美化提示词',
}

const PARAM_LABELS: Record<string, string> = {
  amount: '强度',
  strength: '强度',
  radius_scale: '半径倍率',
  detail_scale: '细节倍率',
  highlight_protection: '高光保护',
  shadow_protection: '阴影保护',
  luminance_protection: '亮度保护',
  color_protection: '颜色保护',
  noise_protection: '噪点保护',
  feather_radius: '羽化',
  whites_amount: '白场',
  blacks_amount: '黑场',
  shadow_lift: '阴影提拉',
  midtone_gamma: '中间调',
  highlight_compress: '高光压缩',
  contrast_bias: '对比偏置',
  orange_saturation: '橙色饱和',
  blue_saturation: '蓝色饱和',
  blue_luminance: '蓝色明度',
  yellow_luminance: '黄色明度',
  max_crop_ratio: '最大裁剪',
  max_straighten_angle: '最大拉直角',
  straighten_bias: '拉直偏置',
  crop_guard: '安全边距',
  min_scale: '最小保留',
  luma_scale: '亮度去噪',
  chroma_scale: '色度去噪',
  detail_protection: '细节保护',
  amount_scale: '锐化量倍率',
  threshold_scale: '阈值倍率',
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === 'object' && !Array.isArray(value)
}

function asString(value: unknown): string | null {
  return typeof value === 'string' && value.length ? value : null
}

function asStringArray(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string') : []
}

function getPackageMeta(op?: string | null): { label: string; description: string } {
  if (!op) {
    return {
      label: '未命名工具',
      description: '没有返回工具标识。',
    }
  }
  return PACKAGE_META[op] ?? {
    label: op.replaceAll('_', ' '),
    description: '当前没有额外说明，展示的是后端返回的工具标识。',
  }
}

function getRegionLabel(region?: string | null): string {
  if (!region) return '未标记区域'
  return REGION_LABELS[region] ?? region
}

function getStageLabel(stage?: string | null): string {
  if (!stage) return '未标记阶段'
  return STAGE_LABELS[stage] ?? stage
}

function getFallbackSourceLabel(source?: string | null): string {
  if (!source) return '未标记来源'
  return FALLBACK_SOURCE_LABELS[source] ?? source
}

function getFallbackStrategyLabel(strategy?: string | null): string {
  if (!strategy) return '未标记策略'
  return FALLBACK_STRATEGY_LABELS[strategy] ?? strategy
}

function getRoundLabel(roundKey: string, fallbackIndex: number): string {
  const matched = roundKey.match(/round_(\d+)/)
  if (matched) return `第 ${matched[1]} 轮`
  return fallbackIndex === 1 ? '本次执行' : `执行分组 ${fallbackIndex}`
}

function getRoundSortValue(roundKey: string, fallbackIndex: number): number {
  const matched = roundKey.match(/round_(\d+)/)
  return matched ? Number(matched[1]) : fallbackIndex
}

function formatValue(value: unknown): string {
  if (typeof value === 'number') {
    if (Number.isInteger(value)) return String(value)
    return value.toFixed(Math.abs(value) >= 10 ? 1 : 2).replace(/\.?0+$/, '')
  }
  if (typeof value === 'boolean') return value ? '是' : '否'
  if (typeof value === 'string') return value
  if (Array.isArray(value)) return `${value.length} 项`
  if (isRecord(value)) return `${Object.keys(value).length} 项`
  return String(value)
}

function formatDuration(durationMs: number): string {
  if (durationMs < 1000) return `${durationMs} ms`
  if (durationMs < 60_000) return `${(durationMs / 1000).toFixed(durationMs >= 10_000 ? 1 : 2).replace(/\.?0+$/, '')} s`
  const minutes = Math.floor(durationMs / 60_000)
  const seconds = ((durationMs % 60_000) / 1000).toFixed(1).replace(/\.0$/, '')
  return `${minutes} min ${seconds} s`
}

function getParamLabel(key: string): string {
  return PARAM_LABELS[key] ?? key.replaceAll('_', ' ')
}

function extractParamBadges(item: ExecutionTraceItem): string[] {
  const appliedParams = isRecord(item.applied_params) ? item.applied_params : null
  const rawParams = appliedParams && isRecord(appliedParams.params) ? appliedParams.params : appliedParams
  if (!rawParams) return []

  return Object.entries(rawParams)
    .filter(([, value]) => {
      if (value == null) return false
      if (typeof value === 'number') return Math.abs(value) > 0.0001
      if (typeof value === 'string') return value.length > 0
      if (typeof value === 'boolean') return value
      if (Array.isArray(value)) return value.length > 0
      if (isRecord(value)) return Object.keys(value).length > 0
      return true
    })
    .slice(0, 6)
    .map(([key, value]) => `${getParamLabel(key)} ${formatValue(value)}`)
}

function extractParamEntries(item: ExecutionTraceItem): ParamEntry[] {
  const appliedParams = isRecord(item.applied_params) ? item.applied_params : null
  const rawParams = appliedParams && isRecord(appliedParams.params) ? appliedParams.params : appliedParams
  if (!rawParams) return []

  return Object.entries(rawParams)
    .filter(([, value]) => {
      if (value == null) return false
      if (typeof value === 'number') return Math.abs(value) > 0.0001
      if (typeof value === 'string') return value.length > 0
      if (typeof value === 'boolean') return value
      if (Array.isArray(value)) return value.length > 0
      if (isRecord(value)) return Object.keys(value).length > 0
      return true
    })
    .map(([key, value]) => ({
      key,
      label: getParamLabel(key),
      value: formatValue(value),
    }))
}

function getOutputAsset(item: ExecutionTraceItem): AssetResponse | null {
  const outputAsset = item.output_asset
  if (isRecord(outputAsset) && typeof outputAsset.content_url === 'string') {
    return outputAsset as AssetResponse
  }
  return null
}

function getStatusTone(item: { ok?: boolean | null; fallbackUsed?: boolean; error?: string | null }): StatusTone {
  if (item.ok === false || item.error) return 'error'
  if (item.fallbackUsed) return 'warning'
  if (item.ok) return 'success'
  return 'neutral'
}

function getStatusLabel(item: { ok?: boolean | null; fallbackUsed?: boolean; error?: string | null }): string {
  if (item.ok === false || item.error) return '失败'
  if (item.fallbackUsed) return '已回退'
  if (item.ok) return '成功'
  return '未知'
}

function getOperationsFromPlan(plan: unknown): Record<string, unknown>[] {
  if (!isRecord(plan)) return []
  const operations = plan.operations
  return Array.isArray(operations) ? operations.filter(isRecord) : []
}

function buildPlannerMaskItems(
  plan: unknown,
  roundKey: string,
  roundLabel: string,
): PlannerMaskItem[] {
  return getOperationsFromPlan(plan)
    .filter((operation) => asString(operation.region) !== 'whole_image')
    .map((operation, itemIndex) => {
      const params = isRecord(operation.params) ? operation.params : {}
      const op = asString(operation.op)
      const meta = getPackageMeta(op)
      return {
        sequence: typeof operation.priority === 'number' ? operation.priority + 1 : itemIndex + 1,
        roundKey,
        roundLabel,
        op,
        opLabel: meta.label,
        opDescription: meta.description,
        region: asString(operation.region),
        regionLabel: getRegionLabel(asString(operation.region)),
        maskProvider: asString(params.mask_provider),
        maskPrompt: asString(params.mask_prompt),
        maskNegativePrompt: asString(params.mask_negative_prompt),
        maskSemanticType: typeof params.mask_semantic_type === 'boolean' ? params.mask_semantic_type : null,
      }
    })
}

const plannerMaskGroups = computed<PlannerMaskGroup[]>(() => {
  const roundEntries = Object.entries(props.jobDetail.round_plans || {}).filter(
    ([, plan]) => buildPlannerMaskItems(plan, 'noop', 'noop').length > 0,
  )

  if (roundEntries.length > 0) {
    return roundEntries
      .map(([roundKey, plan], groupIndex) => ({
        roundKey,
        plan,
        sortValue: getRoundSortValue(roundKey, groupIndex + 1),
      }))
      .sort((left, right) => left.sortValue - right.sortValue)
      .map(({ roundKey, plan }, groupIndex) => {
        const roundLabel = getRoundLabel(roundKey, groupIndex + 1)
        return {
          key: roundKey,
          label: roundLabel,
          items: buildPlannerMaskItems(plan, roundKey, roundLabel),
        }
      })
  }

  const fallbackItems = buildPlannerMaskItems(props.jobDetail.edit_plan, 'full_run', '本次执行')
  return fallbackItems.length
    ? [
        {
          key: 'full_run',
          label: '本次执行',
          items: fallbackItems,
        },
      ]
    : []
})

const traceGroups = computed<TraceGroup[]>(() => {
  const roundEntries = Object.entries(props.jobDetail.round_execution_traces || {}).filter(
    ([, items]) => Array.isArray(items) && items.length > 0,
  )

  if (roundEntries.length > 0) {
    return roundEntries
      .map(([roundKey, items], groupIndex) => ({
        roundKey,
        items,
        sortValue: getRoundSortValue(roundKey, groupIndex + 1),
      }))
      .sort((left, right) => left.sortValue - right.sortValue)
      .map(({ roundKey, items }, groupIndex) => {
        const roundLabel = getRoundLabel(roundKey, groupIndex + 1)
        return {
          key: roundKey,
          label: roundLabel,
          items: items.map((item, itemIndex) => {
            const meta = getPackageMeta(item.op)
            return {
              sequence: (item.index ?? itemIndex) + 1,
              stage: item.stage,
              roundKey,
              roundLabel,
              op: item.op,
              opLabel: meta.label,
              opDescription: meta.description,
              region: item.region,
              regionLabel: getRegionLabel(item.region),
              ok: item.ok,
              fallbackUsed: Boolean(item.fallback_used),
              error: item.error,
              maskPath: typeof item.mask_path === 'string' ? item.mask_path : null,
              paramBadges: extractParamBadges(item),
              paramEntries: extractParamEntries(item),
              outputImageUrl: getOutputAsset(item)?.content_url ?? null,
              outputFilename: getOutputAsset(item)?.filename ?? null,
            }
          }),
        }
      })
  }

  if (!props.jobDetail.execution_trace.length) return []

  return [
    {
      key: 'full_run',
      label: '本次执行',
      items: props.jobDetail.execution_trace.map((item, itemIndex) => {
        const meta = getPackageMeta(item.op)
        return {
          sequence: (item.index ?? itemIndex) + 1,
          stage: item.stage,
          roundKey: 'full_run',
          roundLabel: '本次执行',
          op: item.op,
          opLabel: meta.label,
          opDescription: meta.description,
          region: item.region,
          regionLabel: getRegionLabel(item.region),
          ok: item.ok,
          fallbackUsed: Boolean(item.fallback_used),
          error: item.error,
          maskPath: typeof item.mask_path === 'string' ? item.mask_path : null,
          paramBadges: extractParamBadges(item),
          paramEntries: extractParamEntries(item),
          outputImageUrl: getOutputAsset(item)?.content_url ?? null,
          outputFilename: getOutputAsset(item)?.filename ?? null,
        }
      }),
    },
  ]
})

const segmentationGroups = computed<SegmentationGroup[]>(() => {
  const roundEntries = Object.entries(props.jobDetail.round_segmentation_traces || {}).filter(
    ([, items]) => Array.isArray(items) && items.length > 0,
  )

  if (roundEntries.length > 0) {
    return roundEntries
      .map(([roundKey, items], groupIndex) => ({
        roundKey,
        items,
        sortValue: getRoundSortValue(roundKey, groupIndex + 1),
      }))
      .sort((left, right) => left.sortValue - right.sortValue)
      .map(({ roundKey, items }, groupIndex) => {
        const roundLabel = getRoundLabel(roundKey, groupIndex + 1)
        return {
          key: roundKey,
          label: roundLabel,
          items: items.map((item, itemIndex) => {
            const meta = getPackageMeta(item.source_op)
            return {
              sequence: (item.index ?? itemIndex) + 1,
              roundKey,
              roundLabel,
              sourceOp: item.source_op,
              sourceOpLabel: meta.label,
              region: item.region,
              regionLabel: getRegionLabel(item.region),
              provider: asString(item.provider),
              requestedProvider: asString(item.requested_provider),
              targetLabel: asString(item.target_label),
              prompt: asString(item.prompt),
              negativePrompt: asString(item.negative_prompt),
              semanticType: typeof item.semantic_type === 'boolean' ? item.semantic_type : null,
              ok: item.ok,
              fallbackUsed: Boolean(item.fallback_used),
              error: item.error,
              maskPath: asString(item.mask_path),
              attemptIndex: typeof item.attempt_index === 'number' ? item.attempt_index : null,
              attemptStrategy: asString(item.attempt_strategy),
              requestedPrompt: asString(item.requested_prompt),
              effectivePrompt: asString(item.effective_prompt),
              revertMask: typeof item.revert_mask === 'boolean' ? item.revert_mask : null,
              apiChain: asStringArray(item.api_chain),
            }
          }),
        }
      })
  }

  if (!props.jobDetail.segmentation_trace.length) return []

  return [
    {
      key: 'full_run',
      label: '本次执行',
      items: props.jobDetail.segmentation_trace.map((item, itemIndex) => {
        const meta = getPackageMeta(item.source_op)
        return {
          sequence: (item.index ?? itemIndex) + 1,
          roundKey: 'full_run',
          roundLabel: '本次执行',
          sourceOp: item.source_op,
          sourceOpLabel: meta.label,
          region: item.region,
          regionLabel: getRegionLabel(item.region),
          provider: asString(item.provider),
          requestedProvider: asString(item.requested_provider),
          targetLabel: asString(item.target_label),
          prompt: asString(item.prompt),
          negativePrompt: asString(item.negative_prompt),
          semanticType: typeof item.semantic_type === 'boolean' ? item.semantic_type : null,
          ok: item.ok,
          fallbackUsed: Boolean(item.fallback_used),
          error: item.error,
          maskPath: asString(item.mask_path),
          attemptIndex: typeof item.attempt_index === 'number' ? item.attempt_index : null,
          attemptStrategy: asString(item.attempt_strategy),
          requestedPrompt: asString(item.requested_prompt),
          effectivePrompt: asString(item.effective_prompt),
          revertMask: typeof item.revert_mask === 'boolean' ? item.revert_mask : null,
          apiChain: asStringArray(item.api_chain),
        }
      }),
    },
  ]
})

const flattenedTrace = computed(() => traceGroups.value.flatMap((group) => group.items))
const flattenedSegmentation = computed(() => segmentationGroups.value.flatMap((group) => group.items))
const plannedSegmentationCount = computed(() => plannerMaskGroups.value.reduce((total, group) => total + group.items.length, 0))
const fallbackItems = computed<FallbackViewItem[]>(() => (
  (props.jobDetail.fallback_trace || []).map((item: ApiFallbackTraceItem, index: number) => ({
    key: `${item.stage || 'unknown'}-${item.source || 'unknown'}-${item.location || 'unknown'}-${index}`,
    stageLabel: getStageLabel(typeof item.stage === 'string' ? item.stage : null),
    sourceLabel: getFallbackSourceLabel(typeof item.source === 'string' ? item.source : null),
    locationLabel: typeof item.location === 'string' && item.location.length ? item.location : '未标记位置',
    strategyLabel: getFallbackStrategyLabel(typeof item.strategy === 'string' ? item.strategy : null),
    message: typeof item.message === 'string' && item.message.length ? item.message : '发生了一次自动降级。',
    error: typeof item.error === 'string' ? item.error : null,
  }))
))
const fallbackLocationCount = computed(() => new Set(fallbackItems.value.map((item) => `${item.stageLabel}-${item.locationLabel}`)).size)

const toolSummary = computed<ToolSummaryItem[]>(() => {
  const grouped = new Map<string, ToolSummaryItem>()

  for (const item of flattenedTrace.value) {
    const op = item.op ?? 'unknown'
    const existing = grouped.get(op)
    if (!existing) {
      grouped.set(op, {
        op,
        label: item.opLabel,
        description: item.opDescription,
        count: 1,
        successCount: item.ok ? 1 : 0,
        fallbackCount: item.fallbackUsed ? 1 : 0,
        failureCount: item.ok === false || item.error ? 1 : 0,
        regions: [item.regionLabel],
        rounds: [item.roundLabel],
      })
      continue
    }

    existing.count += 1
    if (item.ok) existing.successCount += 1
    if (item.fallbackUsed) existing.fallbackCount += 1
    if (item.ok === false || item.error) existing.failureCount += 1
    if (!existing.regions.includes(item.regionLabel)) existing.regions.push(item.regionLabel)
    if (!existing.rounds.includes(item.roundLabel)) existing.rounds.push(item.roundLabel)
  }

  return [...grouped.values()].sort((left, right) => right.count - left.count || left.label.localeCompare(right.label))
})

const statCards = computed(() => {
  const total = flattenedTrace.value.length
  const successCount = flattenedTrace.value.filter((item) => item.ok).length
  const fallbackCount = fallbackItems.value.length
  const failureCount = flattenedTrace.value.filter((item) => item.ok === false || item.error).length

  return [
    { label: '工具调用', value: total, tone: 'neutral' as StatusTone },
    { label: '实际工具数', value: toolSummary.value.length, tone: 'neutral' as StatusTone },
    { label: '计划分割', value: plannedSegmentationCount.value, tone: 'neutral' as StatusTone },
    { label: '实际分割', value: flattenedSegmentation.value.length, tone: 'neutral' as StatusTone },
    { label: 'Fallback 次数', value: fallbackCount, tone: fallbackCount > 0 ? 'warning' as StatusTone : 'neutral' as StatusTone },
    { label: 'Fallback 地点', value: fallbackLocationCount.value, tone: fallbackCount > 0 ? 'warning' as StatusTone : 'neutral' as StatusTone },
    { label: '成功执行', value: successCount, tone: 'success' as StatusTone },
    {
      label: '回退/失败',
      value: failureCount,
      tone: failureCount > 0 ? ('error' as StatusTone) : ('warning' as StatusTone),
    },
  ]
})

const stageTimingItems = computed<StageTimingViewItem[]>(() => (
  (props.jobDetail.stage_timings || [])
    .map((item: StageTimingResponse) => ({
      key: `${item.stage}-${item.started_at}`,
      label: item.label || getStageLabel(item.stage),
      durationLabel: formatDuration(item.duration_ms),
      tone: item.status === 'failed' ? 'error' as StatusTone : 'success' as StatusTone,
      meta: [
        item.stage,
        item.status === 'failed' ? '失败结束' : '完成',
      ],
    }))
))

function buildTimelineEntry(event: JobEvent, index: number): TimelineEntry | null {
  if (!event.message) return null
  const meta: string[] = []
  if (event.round) meta.push(getRoundLabel(event.round, index + 1))
  if (event.op) meta.push(getPackageMeta(event.op).label)
  else if (event.node) meta.push(getStageLabel(event.node))
  else if (event.stage) meta.push(getStageLabel(event.stage))
  if (event.region) meta.push(getRegionLabel(event.region))
  if (typeof event.provider === 'string') meta.push(event.provider)

  const tone: StatusTone =
    event.event.includes('failed') ? 'error' :
      event.event.includes('interrupt') ? 'warning' :
        event.event.includes('completed') || event.event.includes('finished') ? 'success' :
          'neutral'

  let title = '流程事件'
  if (event.event.startsWith('segmentation_')) {
    title =
      event.event === 'segmentation_started' ? '开始分割' :
        event.event === 'segmentation_finished' ? '分割完成' :
          event.event === 'segmentation_skipped' ? '分割跳过' :
          '分割失败'
  } else if (event.event.startsWith('planner_')) {
    title =
      event.event === 'planner_started' ? '规划开始' :
        event.event === 'planner_tool_called' ? '规划选择工具' :
          event.event === 'planner_tool_resolved' ? '规划工具纠正' :
          event.event === 'planner_tool_finished' ? '规划工具完成' :
            event.event === 'planner_tool_failed' ? '规划工具失败' :
              '规划轮结束'
  } else if (event.event.startsWith('bootstrap_')) {
    title =
      event.event === 'bootstrap_started' ? 'Bootstrap 开始' :
        event.event === 'bootstrap_finished' ? 'Bootstrap 完成' :
          'Bootstrap 回退'
  } else if (event.op) {
    title = `${getPackageMeta(event.op).label}${event.event === 'package_started' ? '开始执行' : event.event === 'package_finished' ? '执行完成' : event.event === 'package_failed' ? '执行失败' : event.event === 'package_skipped' ? '已跳过' : ''}`
  } else if (event.node) {
    title = getStageLabel(event.node)
  } else if (event.round) {
    title = getRoundLabel(event.round, index + 1)
  } else if (event.event === 'job_created') {
    title = '任务创建'
  } else if (event.event === 'job_completed') {
    title = '任务完成'
  } else if (event.event === 'job_failed') {
    title = '任务失败'
  }

  return {
    key: `${event.event}-${index}`,
    title,
    message: event.message,
    tone,
    meta,
  }
}

const timelineEntries = computed<TimelineEntry[]>(() => (
  props.jobDetail.events
    .filter((event) => [
      'job_created',
      'bootstrap_started',
      'bootstrap_finished',
      'bootstrap_failed',
      'round_started',
      'round_completed',
      'node_started',
      'node_finished',
      'node_failed',
      'planner_started',
      'planner_tool_called',
      'planner_tool_resolved',
      'planner_tool_finished',
      'planner_tool_failed',
      'planner_round_finished',
      'segmentation_started',
      'segmentation_finished',
      'segmentation_skipped',
      'segmentation_failed',
      'package_started',
      'package_finished',
      'package_skipped',
      'package_failed',
      'interrupt',
      'job_interrupted',
      'job_completed',
      'job_failed',
    ].includes(event.event))
    .map((event, index) => buildTimelineEntry(event, index))
    .filter((item): item is TimelineEntry => item !== null)
))
</script>

<template>
  <div class="execution-summary">
    <section class="glass-panel summary-panel">
      <div class="section-head">
        <div>
          <h3>本次调用的工具</h3>
          <p>这里展示的是后端实际执行过的工具，而不是仅仅计划中的操作。</p>
        </div>
      </div>

      <div class="stat-grid" v-if="statCards.length">
        <div
          v-for="card in statCards"
          :key="card.label"
          class="stat-card"
          :class="`tone-${card.tone}`"
        >
          <span class="stat-label">{{ card.label }}</span>
          <strong class="stat-value">{{ card.value }}</strong>
        </div>
      </div>

      <div v-if="toolSummary.length" class="tool-grid">
        <article v-for="tool in toolSummary" :key="tool.op" class="tool-card">
          <div class="tool-card-top">
            <div>
              <h4>{{ tool.label }}</h4>
              <code>{{ tool.op }}</code>
            </div>
            <span class="count-pill">{{ tool.count }} 次</span>
          </div>
          <p class="tool-description">{{ tool.description }}</p>
          <div class="tool-card-meta">
            <span>区域：{{ tool.regions.join(' / ') }}</span>
            <span>轮次：{{ tool.rounds.join(' / ') }}</span>
          </div>
          <div class="tool-card-status">
            <span class="status-chip tone-success">成功 {{ tool.successCount }}</span>
            <span v-if="tool.fallbackCount" class="status-chip tone-warning">回退 {{ tool.fallbackCount }}</span>
            <span v-if="tool.failureCount" class="status-chip tone-error">失败 {{ tool.failureCount }}</span>
          </div>
        </article>
      </div>

      <div v-else class="empty-state">
        当前没有可展示的工具执行记录。
      </div>
    </section>

    <section class="glass-panel stage-panel" v-if="showDebug && stageTimingItems.length">
      <div class="section-head">
        <div>
          <h3>阶段耗时</h3>
          <p>这里按流程阶段展示从开始到结束的实际耗时，方便观察慢点在哪里。</p>
        </div>
      </div>

      <div class="tool-grid">
        <article v-for="item in stageTimingItems" :key="item.key" class="tool-card">
          <div class="tool-card-top">
            <div>
              <h4>{{ item.label }}</h4>
              <code>{{ item.meta[0] }}</code>
            </div>
            <span class="status-chip" :class="`tone-${item.tone}`">{{ item.durationLabel }}</span>
          </div>
          <div class="tool-card-meta">
            <span v-for="meta in item.meta.slice(1)" :key="meta">{{ meta }}</span>
          </div>
        </article>
      </div>
    </section>

    <section class="glass-panel stage-panel" v-if="showDebug && fallbackItems.length">
      <div class="section-head">
        <div>
          <h3>Fallback 统计</h3>
          <p>这里汇总所有自动降级的次数、发生位置和采用的兜底策略。</p>
        </div>
      </div>

      <div class="round-list">
        <article v-for="item in fallbackItems" :key="item.key" class="tool-card">
          <div class="tool-card-top">
            <div>
              <h4>{{ item.message }}</h4>
              <code>{{ item.locationLabel }}</code>
            </div>
            <span class="status-chip tone-warning">{{ item.strategyLabel }}</span>
          </div>
          <div class="tool-card-meta">
            <span>阶段：{{ item.stageLabel }}</span>
            <span>来源：{{ item.sourceLabel }}</span>
            <span>位置：{{ item.locationLabel }}</span>
          </div>
          <p v-if="item.error" class="trace-error">{{ item.error }}</p>
        </article>
      </div>
    </section>

    <section class="glass-panel planner-panel" v-if="showTrace && plannerMaskGroups.length">
      <div class="section-head">
        <div>
          <h3>Planner 分割计划</h3>
          <p>这里展示的是 planner 为每个局部操作给出的区域标签，以及对应的动态分割 prompt。</p>
        </div>
      </div>

      <div class="round-list">
        <section v-for="group in plannerMaskGroups" :key="group.key" class="round-card">
          <div class="round-head">
            <div>
              <h4>{{ group.label }}</h4>
              <p>{{ group.items.length }} 次计划分割</p>
            </div>
          </div>

          <ol class="trace-list">
            <li v-for="item in group.items" :key="`${group.key}-plan-${item.sequence}-${item.op}`" class="trace-item">
              <div class="trace-seq">{{ item.sequence }}</div>
              <div class="trace-body">
                <div class="trace-top">
                  <div>
                    <strong>{{ item.opLabel }}</strong>
                    <code>{{ item.op }}</code>
                  </div>
                  <span class="status-chip tone-neutral">
                    计划分割
                  </span>
                </div>

                <p class="trace-description">{{ item.opDescription }}</p>

                <div class="trace-meta">
                  <span>区域标签：{{ item.regionLabel }}</span>
                  <span v-if="item.maskProvider">分割提供方：{{ item.maskProvider }}</span>
                  <span v-if="item.maskSemanticType">语义分割</span>
                </div>

                <div class="mask-copy">
                  <div class="mask-row">
                    <span>分割目标</span>
                    <strong>{{ item.maskPrompt || '当前计划未返回 prompt' }}</strong>
                  </div>
                  <div v-if="item.maskNegativePrompt" class="mask-row">
                    <span>排除区域</span>
                    <strong>{{ item.maskNegativePrompt }}</strong>
                  </div>
                </div>
              </div>
            </li>
          </ol>
        </section>
      </div>
    </section>

    <section class="glass-panel segmentation-panel" v-if="showTrace && segmentationGroups.length">
      <div class="section-head">
        <div>
          <h3>实际分割明细</h3>
          <p>这里只统计真正发起过的分割请求，缓存命中不会重复计数。</p>
        </div>
      </div>

      <div class="round-list">
        <section v-for="group in segmentationGroups" :key="group.key" class="round-card">
          <div class="round-head">
            <div>
              <h4>{{ group.label }}</h4>
              <p>{{ group.items.length }} 次实际分割</p>
            </div>
          </div>

          <ol class="trace-list">
            <li v-for="item in group.items" :key="`${group.key}-seg-${item.sequence}-${item.sourceOp}`" class="trace-item">
              <div class="trace-seq">{{ item.sequence }}</div>
              <div class="trace-body">
                <div class="trace-top">
                  <div>
                    <strong>{{ item.targetLabel || item.prompt || '未命名分割目标' }}</strong>
                    <code>{{ item.sourceOp || 'unknown_op' }}</code>
                  </div>
                  <span class="status-chip" :class="`tone-${getStatusTone(item)}`">
                    {{ getStatusLabel(item) }}
                  </span>
                </div>

                <p class="trace-description">
                  来源工具：{{ item.sourceOpLabel }}。这次分割最终命中的目标是
                  {{ item.targetLabel || item.prompt || '未返回具体描述' }}。
                </p>

                <div class="trace-meta">
                  <span>区域标签：{{ item.regionLabel }}</span>
                  <span v-if="item.requestedProvider && item.provider && item.requestedProvider !== item.provider">
                    {{ item.requestedProvider }} -> {{ item.provider }}
                  </span>
                  <span v-else-if="item.provider">提供方：{{ item.provider }}</span>
                  <span v-if="item.semanticType">语义分割</span>
                  <span v-if="item.maskPath">已生成区域遮罩</span>
                  <span v-if="typeof item.attemptIndex === 'number'">尝试：第 {{ item.attemptIndex + 1 }} 次</span>
                  <span v-if="item.attemptStrategy">策略：{{ item.attemptStrategy }}</span>
                </div>

                <div class="mask-copy">
                  <div v-if="item.prompt" class="mask-row">
                    <span>Prompt</span>
                    <strong>{{ item.prompt }}</strong>
                  </div>
                  <div v-if="item.requestedPrompt && item.requestedPrompt !== item.prompt" class="mask-row">
                    <span>原始目标</span>
                    <strong>{{ item.requestedPrompt }}</strong>
                  </div>
                  <div v-if="item.effectivePrompt && item.effectivePrompt !== item.prompt" class="mask-row">
                    <span>实际尝试</span>
                    <strong>{{ item.effectivePrompt }}</strong>
                  </div>
                  <div v-if="typeof item.revertMask === 'boolean'" class="mask-row">
                    <span>反向遮罩</span>
                    <strong>{{ item.revertMask ? '是' : '否' }}</strong>
                  </div>
                  <div v-if="item.negativePrompt" class="mask-row">
                    <span>Negative Prompt</span>
                    <strong>{{ item.negativePrompt }}</strong>
                  </div>
                  <div v-if="item.apiChain.length" class="mask-row">
                    <span>调用链</span>
                    <strong>{{ item.apiChain.join(' -> ') }}</strong>
                  </div>
                </div>

                <p v-if="item.error" class="trace-error">{{ item.error }}</p>
              </div>
            </li>
          </ol>
        </section>
      </div>
    </section>

    <section class="glass-panel details-panel" v-if="showTrace && traceGroups.length">
      <div class="section-head">
        <div>
          <h3>执行明细</h3>
          <p>按轮次展开每一次实际调用，能看到工具、作用区域、参数和执行状态。</p>
        </div>
      </div>

      <div class="round-list">
        <section v-for="group in traceGroups" :key="group.key" class="round-card">
          <div class="round-head">
            <div>
              <h4>{{ group.label }}</h4>
              <p>{{ group.items.length }} 个工具调用</p>
            </div>
          </div>

          <ol class="trace-list">
            <li v-for="item in group.items" :key="`${group.key}-${item.sequence}-${item.op}`" class="trace-item">
              <div class="trace-seq">{{ item.sequence }}</div>
              <div class="trace-body">
                <div class="trace-top">
                  <div>
                    <strong>{{ item.opLabel }}</strong>
                    <code>{{ item.op }}</code>
                  </div>
                  <span class="status-chip" :class="`tone-${getStatusTone(item)}`">
                    {{ getStatusLabel(item) }}
                  </span>
                </div>

                <p class="trace-description">{{ item.opDescription }}</p>

                <div class="trace-meta">
                  <span>区域：{{ item.regionLabel }}</span>
                  <span v-if="item.stage">阶段：{{ getStageLabel(item.stage) }}</span>
                  <span v-if="item.maskPath">使用了区域遮罩</span>
                </div>

                <div v-if="item.paramBadges.length" class="param-badges">
                  <span v-for="badge in item.paramBadges" :key="badge" class="param-badge">
                    {{ badge }}
                  </span>
                </div>

                <div v-if="item.paramEntries.length" class="trace-params">
                  <div class="trace-section-title">本次入参</div>
                  <dl class="param-list">
                    <div v-for="entry in item.paramEntries" :key="`${item.roundKey}-${item.sequence}-${entry.key}`" class="param-row">
                      <dt>{{ entry.label }}</dt>
                      <dd>{{ entry.value }}</dd>
                    </div>
                  </dl>
                </div>

                <div v-if="item.outputImageUrl" class="trace-preview">
                  <div class="trace-section-title">本次工具输出</div>
                  <div class="trace-image-card">
                    <img :src="item.outputImageUrl" :alt="item.outputFilename || `${item.opLabel} output`" class="trace-image" />
                    <div class="trace-image-meta">
                      <span>调用后图片</span>
                      <strong>{{ item.outputFilename || '生成结果' }}</strong>
                    </div>
                  </div>
                </div>

                <p v-if="item.error" class="trace-error">{{ item.error }}</p>
              </div>
            </li>
          </ol>
        </section>
      </div>
    </section>

    <section class="glass-panel timeline-panel" v-if="showTrace && timelineEntries.length">
      <div class="section-head">
        <div>
          <h3>流程记录</h3>
          <p>如果你想回看整个过程，下面是从创建任务到完成的关键事件。</p>
        </div>
      </div>

      <ul class="timeline-list">
        <li v-for="entry in timelineEntries" :key="entry.key" class="timeline-item">
          <span class="timeline-dot" :class="`tone-${entry.tone}`"></span>
          <div class="timeline-body">
            <div class="timeline-title-row">
              <strong>{{ entry.title }}</strong>
              <div class="timeline-meta" v-if="entry.meta.length">
                <span v-for="meta in entry.meta" :key="meta">{{ meta }}</span>
              </div>
            </div>
            <p>{{ entry.message }}</p>
          </div>
        </li>
      </ul>
    </section>
  </div>
</template>

<style scoped>
.execution-summary {
  display: flex;
  flex-direction: column;
  gap: 24px;
}

.summary-panel,
.stage-panel,
.planner-panel,
.segmentation-panel,
.details-panel,
.timeline-panel {
  padding: 24px;
}

.section-head {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 16px;
  margin-bottom: 20px;
}

.section-head h3 {
  margin: 0 0 6px;
  font-size: 1.1rem;
}

.section-head p {
  margin: 0;
  color: var(--text-muted);
  font-size: 0.95rem;
}

.stat-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 14px;
  margin-bottom: 20px;
}

.stat-card {
  padding: 16px;
  border-radius: 16px;
  border: 1px solid var(--border-glass);
  background: rgba(255, 255, 255, 0.03);
}

.stat-label {
  display: block;
  font-size: 0.85rem;
  color: var(--text-muted);
  margin-bottom: 8px;
}

.stat-value {
  font-size: 1.55rem;
  line-height: 1;
}

.tool-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px;
}

.tool-card {
  padding: 18px;
  border-radius: 18px;
  border: 1px solid var(--border-glass);
  background: rgba(255, 255, 255, 0.03);
}

.tool-card-top {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  align-items: flex-start;
}

.tool-card-top h4,
.round-head h4 {
  margin: 0 0 6px;
  font-size: 1rem;
}

.tool-card-top code,
.trace-top code {
  color: #c7d2fe;
  font-size: 0.8rem;
}

.count-pill {
  white-space: nowrap;
  padding: 6px 10px;
  border-radius: 999px;
  background: rgba(99, 102, 241, 0.16);
  color: #c7d2fe;
  font-size: 0.8rem;
}

.tool-description,
.trace-description {
  margin: 12px 0 0;
  color: var(--text-muted);
  font-size: 0.92rem;
  line-height: 1.55;
}

.tool-card-meta,
.tool-card-status,
.trace-meta,
.param-badges,
.timeline-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.tool-card-meta {
  margin-top: 14px;
}

.tool-card-meta span,
.trace-meta span,
.timeline-meta span {
  padding: 5px 10px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.05);
  color: var(--text-muted);
  font-size: 0.78rem;
}

.tool-card-status {
  margin-top: 12px;
}

.status-chip {
  display: inline-flex;
  align-items: center;
  padding: 5px 10px;
  border-radius: 999px;
  font-size: 0.78rem;
  border: 1px solid transparent;
}

.tone-success {
  color: #a7f3d0;
  background: rgba(16, 185, 129, 0.14);
  border-color: rgba(16, 185, 129, 0.22);
}

.tone-warning {
  color: #fde68a;
  background: rgba(245, 158, 11, 0.14);
  border-color: rgba(245, 158, 11, 0.22);
}

.tone-error {
  color: #fca5a5;
  background: rgba(239, 68, 68, 0.14);
  border-color: rgba(239, 68, 68, 0.22);
}

.tone-neutral {
  color: #d1d5db;
  background: rgba(255, 255, 255, 0.06);
  border-color: rgba(255, 255, 255, 0.08);
}

.round-list {
  display: flex;
  flex-direction: column;
  gap: 18px;
}

.round-card {
  padding: 18px;
  border-radius: 18px;
  border: 1px solid var(--border-glass);
  background: rgba(255, 255, 255, 0.03);
}

.round-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
  margin-bottom: 16px;
}

.round-head p {
  margin: 0;
  color: var(--text-muted);
  font-size: 0.88rem;
}

.trace-list {
  list-style: none;
  padding: 0;
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.trace-item {
  display: grid;
  grid-template-columns: 42px 1fr;
  gap: 14px;
  align-items: flex-start;
}

.trace-seq {
  width: 42px;
  height: 42px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 14px;
  background: rgba(99, 102, 241, 0.12);
  color: #c7d2fe;
  font-weight: 700;
}

.trace-body {
  padding: 14px;
  border-radius: 16px;
  border: 1px solid rgba(255, 255, 255, 0.06);
  background: rgba(0, 0, 0, 0.14);
}

.trace-top {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  align-items: flex-start;
}

.trace-top strong {
  display: block;
  margin-bottom: 6px;
}

.trace-meta {
  margin-top: 12px;
}

.param-badges,
.mask-copy {
  margin-top: 12px;
}

.trace-params,
.trace-preview {
  margin-top: 14px;
}

.trace-section-title {
  margin-bottom: 10px;
  color: #cbd5f5;
  font-size: 0.82rem;
  font-weight: 700;
  letter-spacing: 0.02em;
}

.param-badge {
  padding: 6px 10px;
  border-radius: 10px;
  background: rgba(255, 255, 255, 0.06);
  border: 1px solid rgba(255, 255, 255, 0.08);
  color: var(--text-main);
  font-size: 0.8rem;
}

.param-list {
  margin: 0;
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px 12px;
}

.param-row {
  display: grid;
  grid-template-columns: minmax(0, 108px) minmax(0, 1fr);
  gap: 10px;
  align-items: start;
  padding: 10px 12px;
  border-radius: 12px;
  background: rgba(255, 255, 255, 0.04);
  border: 1px solid rgba(255, 255, 255, 0.06);
}

.param-row dt {
  margin: 0;
  color: var(--text-muted);
  font-size: 0.78rem;
}

.param-row dd {
  margin: 0;
  color: var(--text-main);
  font-size: 0.86rem;
  line-height: 1.5;
  word-break: break-word;
}

.mask-copy {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.mask-row {
  display: grid;
  grid-template-columns: 90px 1fr;
  gap: 12px;
  align-items: flex-start;
  padding: 10px 12px;
  border-radius: 12px;
  background: rgba(255, 255, 255, 0.04);
  border: 1px solid rgba(255, 255, 255, 0.06);
}

.mask-row span {
  color: var(--text-muted);
  font-size: 0.78rem;
}

.mask-row strong {
  color: var(--text-main);
  font-size: 0.88rem;
  line-height: 1.5;
  word-break: break-word;
}

.trace-image-card {
  overflow: hidden;
  border-radius: 16px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  background: rgba(255, 255, 255, 0.03);
}

.trace-image {
  width: 100%;
  max-height: 260px;
  object-fit: contain;
  display: block;
  background:
    linear-gradient(135deg, rgba(99, 102, 241, 0.08), rgba(15, 23, 42, 0.2)),
    radial-gradient(circle at top left, rgba(255, 255, 255, 0.06), transparent 48%);
}

.trace-image-meta {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  padding: 12px 14px;
  border-top: 1px solid rgba(255, 255, 255, 0.06);
  color: var(--text-muted);
  font-size: 0.8rem;
}

.trace-image-meta strong {
  color: var(--text-main);
  text-align: right;
  word-break: break-word;
}

.trace-error {
  margin: 12px 0 0;
  padding: 10px 12px;
  border-radius: 12px;
  background: rgba(127, 29, 29, 0.18);
  border: 1px solid rgba(248, 113, 113, 0.18);
  color: #fecaca;
  font-size: 0.84rem;
  line-height: 1.5;
}

.timeline-list {
  list-style: none;
  padding: 0;
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.timeline-item {
  display: grid;
  grid-template-columns: 18px 1fr;
  gap: 14px;
  align-items: flex-start;
}

.timeline-dot {
  width: 12px;
  height: 12px;
  border-radius: 999px;
  margin-top: 5px;
}

.timeline-body {
  padding: 14px 16px;
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.03);
  border: 1px solid var(--border-glass);
}

.timeline-title-row {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  align-items: flex-start;
  margin-bottom: 8px;
}

.timeline-body p {
  margin: 0;
  color: var(--text-muted);
  line-height: 1.55;
}

.empty-state {
  padding: 18px;
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.03);
  color: var(--text-muted);
}

@media (max-width: 960px) {
  .tool-grid {
    grid-template-columns: 1fr;
  }

  .param-list {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 720px) {
  .summary-panel,
  .stage-panel,
  .planner-panel,
  .segmentation-panel,
  .details-panel,
  .timeline-panel {
    padding: 18px;
  }

  .trace-item,
  .timeline-item {
    grid-template-columns: 1fr;
  }

  .trace-seq {
    width: 36px;
    height: 36px;
  }

  .trace-top,
  .timeline-title-row,
  .tool-card-top,
  .round-head,
  .mask-row {
    flex-direction: column;
    grid-template-columns: 1fr;
  }
}
</style>
