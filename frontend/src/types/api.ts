export type JobStatus = 'pending' | 'running' | 'completed' | 'failed' | 'review_required'

export interface AssetResponse {
  asset_id: string
  filename: string
  media_type?: string | null
  size_bytes?: number | null
  created_at: string
  content_url: string
}

export interface UploadAssetsResponse {
  items: AssetResponse[]
}

export interface EditRequest {
  user_id: string
  thread_id?: string | null
  instruction?: string | null
  auto_mode?: boolean
  planner_thinking_mode?: boolean
  input_asset_ids?: string[]
  input_image_paths?: string[]
}

export interface ErrorDetail {
  type: string
  message: string
  stage?: string | null
  node?: string | null
  op?: string | null
  region?: string | null
  traceback?: string | null
  [key: string]: unknown
}

export interface JobSummaryResponse {
  job_id: string
  status: JobStatus
  user_id: string
  thread_id: string
  created_at: string
  updated_at: string
  approval_required: boolean
  request_text?: string | null
  current_stage?: string | null
  current_message?: string | null
  error?: string | null
  error_detail?: ErrorDetail | null
}

export interface StageTimingResponse {
  stage: string
  label: string
  started_at: string
  ended_at: string
  duration_ms: number
  duration_seconds: number
  status: 'completed' | 'failed'
}

export interface ExecutionTraceItem {
  index?: number | null
  stage?: string | null
  op?: string | null
  region?: string | null
  ok?: boolean | null
  fallback_used?: boolean
  error?: string | null
  output_image?: string | null
  output_asset_id?: string | null
  output_asset?: AssetResponse | null
  applied_params?: Record<string, unknown> | null
  mask_path?: string | null
  warnings?: string[] | null
  artifacts?: Record<string, unknown> | null
  [key: string]: unknown
}

export interface EditOperation {
  op: string
  region: string
  strength?: number | null
  params: Record<string, unknown>
  constraints: string[]
  priority: number
}

export interface EditPlan {
  mode: 'explicit' | 'auto'
  domain: 'portrait' | 'landscape' | 'food' | 'document' | 'general'
  executor: 'deterministic' | 'generative' | 'hybrid'
  preserve: string[]
  operations: EditOperation[]
  should_write_memory: boolean
  memory_candidates: Record<string, unknown>[]
  needs_confirmation: boolean
}

export interface PlannerExecutionPlan {
  mode: 'explicit' | 'auto'
  domain: 'portrait' | 'landscape' | 'food' | 'document' | 'general'
  executor: 'deterministic' | 'generative' | 'hybrid'
  preserve: string[]
  steps: EditOperation[]
  step_budget: number
  summary: string
  should_write_memory: boolean
  memory_candidates: Record<string, unknown>[]
  needs_confirmation: boolean
}

export interface StageSummary {
  stage: 'technical_prep' | 'global_base' | 'local_balance' | 'subject_refine' | 'finish_output'
  summary: string
  used_tools: string[]
  key_changes: string[]
  remaining_issues: string[]
}

export interface EvaluationReport {
  selected_output?: string | null
  num_operations: number
  success_count: number
  failure_count: number
  fallback_count: number
  has_output: boolean
  overall_ok?: boolean | null
  preserve_ok?: boolean | null
  style_ok?: boolean | null
  artifact_ok?: boolean | null
  issues: string[]
  warnings: string[]
  summary: string
  should_continue_editing: boolean
  should_request_review: boolean
}

export interface FeedbackItem {
  accepted: boolean
  rating?: number | null
  feedback_text?: string | null
  manual_adjustments: Record<string, unknown>
}

export interface SegmentationTraceItem {
  index?: number | null
  stage?: string | null
  source_op?: string | null
  region?: string | null
  provider?: string | null
  requested_provider?: string | null
  target_label?: string | null
  prompt?: string | null
  negative_prompt?: string | null
  semantic_type?: boolean | null
  ok?: boolean | null
  fallback_used?: boolean
  error?: string | null
  mask_path?: string | null
  mask_asset_id?: string | null
  mask_asset?: AssetResponse | null
  preview_path?: string | null
  preview_asset_id?: string | null
  preview_asset?: AssetResponse | null
  request_id?: string | null
  api_chain?: string[] | null
  attempt_index?: number | null
  attempt_strategy?: string | null
  requested_prompt?: string | null
  effective_prompt?: string | null
  revert_mask?: boolean | null
  attempts?: Record<string, unknown>[] | null
  [key: string]: unknown
}

export interface FallbackTraceItem {
  index?: number | null
  stage?: string | null
  source?: string | null
  location?: string | null
  strategy?: string | null
  message?: string
  error?: string | null
  fallback_used?: boolean
  [key: string]: unknown
}

export interface JobEvent {
  event: string
  occurred_at?: string
  stage?: string
  round?: string
  node?: string
  op?: string
  region?: string
  provider?: string
  requested_provider?: string
  prompt?: string
  negative_prompt?: string
  target_label?: string
  message?: string
  job_id?: string
  ok?: boolean
  error?: string | null
  error_detail?: ErrorDetail | null
  payload?: Record<string, unknown> | null
  approval_payload?: Record<string, unknown> | null
  interrupt_id?: string
  [key: string]: unknown
}

export interface PhaseResponse {
  plan?: PlannerExecutionPlan | null
  execution_trace: ExecutionTraceItem[]
  segmentation_trace: SegmentationTraceItem[]
  eval_report?: EvaluationReport | null
  output?: AssetResponse | null
  summary?: StageSummary | null
  skipped: boolean
  skip_reason?: string | null
  trigger_reasons: string[]
  stopped_early: boolean
}

export interface EditResponse {
  job: JobSummaryResponse
  selected_output?: AssetResponse | null
  candidate_outputs: AssetResponse[]
  edit_plan?: EditPlan | null
  eval_report?: EvaluationReport | null
  execution_trace: ExecutionTraceItem[]
  segmentation_trace: SegmentationTraceItem[]
  fallback_trace: FallbackTraceItem[]
  phases: Record<string, PhaseResponse>
  events: JobEvent[]
  stage_timings: StageTimingResponse[]
}

export interface JobDetailResponse {
  job: JobSummaryResponse
  input_assets: AssetResponse[]
  selected_output?: AssetResponse | null
  candidate_outputs: AssetResponse[]
  edit_plan?: EditPlan | null
  eval_report?: EvaluationReport | null
  execution_trace: ExecutionTraceItem[]
  segmentation_trace: SegmentationTraceItem[]
  fallback_trace: FallbackTraceItem[]
  phases: Record<string, PhaseResponse>
  events: JobEvent[]
  stage_timings: StageTimingResponse[]
  feedback: FeedbackItem[]
}

export interface FeedbackRequest {
  job_id: string
  accepted: boolean
  rating?: number | null
  feedback_text?: string | null
  manual_adjustments?: Record<string, unknown>
}

export interface ResumeReviewRequest {
  job_id: string
  approved: boolean
  note?: string | null
}

export interface ResumeReviewResponse {
  job_id: string
  accepted: boolean
  implemented: boolean
  status: JobStatus
  message: string
}

export interface ToolCatalogItem {
  name: string
  label?: string
  description: string
  family?: string
  stage_affinity?: string[]
  supports_mask?: boolean
  supports_whole_image?: boolean
  default_params?: Record<string, unknown>
  planner_schema?: Record<string, unknown>
  primary_param?: string
  supported_regions: string[]
  mask_policy: 'none' | 'optional' | 'required'
  supported_domains: string[]
  risk_level: 'low' | 'medium' | 'high'
  params_schema: Record<string, unknown>
}

export interface ToolCatalogResponse {
  items: ToolCatalogItem[]
}

export interface SseEventPayload {
  event: string
  occurred_at?: string
  stage?: string
  round?: string
  node?: string
  op?: string
  region?: string
  provider?: string
  requested_provider?: string
  prompt?: string
  negative_prompt?: string
  target_label?: string
  message?: string
  job_id?: string
  error?: string
  error_detail?: Record<string, unknown>
  payload?: Record<string, unknown>
  [key: string]: unknown
}
