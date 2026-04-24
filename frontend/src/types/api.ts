export type JobStatus = 'pending' | 'running' | 'completed' | 'failed' | 'review_required'
export type FocusKey = 'global_tone' | 'subject_separation' | 'subject_cleanup' | 'finish'
export type RoundAction = 'keep' | 'recover_same_round' | 'stop_round'

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
  node?: string | null
  round?: string | null
  focus?: FocusKey | null
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
  current_round?: string | null
  current_focus?: FocusKey | null
  current_message?: string | null
  error?: string | null
  error_detail?: ErrorDetail | null
}

export interface RoundTimingResponse {
  round: string
  focus?: FocusKey | null
  label: string
  started_at: string
  ended_at: string
  duration_ms: number
  duration_seconds: number
  status: 'completed' | 'failed'
}

export interface ExecutionTraceItem {
  index?: number | null
  round_id?: string | null
  focus?: FocusKey | null
  candidate_id?: string | null
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

export interface ObjectiveGap {
  id: string
  focus: FocusKey
  description: string
  priority: number
  target_region?: string
  desired_delta?: string
  constraints?: string[]
  resolved?: boolean
}

export interface RequestGoal {
  kind: string
  target_region: string
  priority: number
  intensity?: number | null
  constraints: string[]
  source: 'heuristic' | 'model' | 'explicit_tool'
}

export interface ObjectiveCard {
  mode: 'auto' | 'explicit'
  domain: string
  summary: string
  goals: RequestGoal[]
  gaps: ObjectiveGap[]
  preserve: string[]
  constraints: string[]
}

export interface CandidateProgram {
  id: string
  label: string
  focus: FocusKey
  source: 'model' | 'rule' | 'variant' | 'noop' | 'direct'
  summary: string
  steps: EditOperation[]
  is_recovery?: boolean
}

export interface CandidatePreviewExecution {
  input_image_path?: string | null
  output_image_path?: string | null
  output_asset_id?: string | null
  output_asset?: AssetResponse | null
  execution_trace: ExecutionTraceItem[]
  segmentation_trace: SegmentationTraceItem[]
  fallback_trace: FallbackTraceItem[]
}

export interface CandidateReview {
  overall_ok: boolean
  preserve_ok: boolean
  style_ok: boolean
  artifact_ok: boolean
  issues: string[]
  warnings: string[]
  summary: string
  recommended_action: RoundAction
  score: number
}

export interface RoundReview {
  overall_ok: boolean
  issues: string[]
  warnings: string[]
  summary: string
  recommended_action: RoundAction
  score: number
}

export interface RecoveryDecision {
  triggered: boolean
  source: 'candidate_review' | 'round_review' | 'deterministic' | 'none'
  fallback_aware: boolean
  reason: string
  candidate_ids: string[]
  selected_candidate_id?: string | null
}

export interface SearchCandidateResponse {
  candidate_id: string
  label: string
  focus: FocusKey
  selected: boolean
  eliminated_reason?: string | null
  program?: CandidateProgram | null
  preview_execution?: CandidatePreviewExecution | null
  review?: CandidateReview | null
}

export interface SearchRoundResponse {
  id: string
  index: number
  focus: FocusKey
  input_image_path?: string | null
  output_image_path?: string | null
  output_asset_id?: string | null
  output_asset?: AssetResponse | null
  objective_gaps: ObjectiveGap[]
  candidates: SearchCandidateResponse[]
  selected_candidate_id?: string | null
  selected_full_execution?: CandidatePreviewExecution | null
  round_review?: RoundReview | null
  recovery_decision?: RecoveryDecision | null
  recovery_candidates: SearchCandidateResponse[]
  completed: boolean
}

export interface SegmentationTraceItem {
  index?: number | null
  round_id?: string | null
  focus?: FocusKey | null
  candidate_id?: string | null
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
  quality_score?: number | null
  quality_flags?: string[] | null
  rejected?: boolean
  [key: string]: unknown
}

export interface FallbackTraceItem {
  index?: number | null
  round_id?: string | null
  focus?: FocusKey | null
  candidate_id?: string | null
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
  round?: string
  focus?: FocusKey
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

export interface EditResponse {
  job: JobSummaryResponse
  selected_output?: AssetResponse | null
  candidate_outputs: AssetResponse[]
  edit_plan?: EditPlan | null
  eval_report?: EvaluationReport | null
  execution_trace: ExecutionTraceItem[]
  segmentation_trace: SegmentationTraceItem[]
  fallback_trace: FallbackTraceItem[]
  objective_card?: ObjectiveCard | null
  rounds: SearchRoundResponse[]
  selected_candidate_id?: string | null
  final_review?: EvaluationReport | null
  final_execution_trace: ExecutionTraceItem[]
  events: JobEvent[]
  round_timings: RoundTimingResponse[]
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
  objective_card?: ObjectiveCard | null
  rounds: SearchRoundResponse[]
  selected_candidate_id?: string | null
  final_review?: EvaluationReport | null
  final_execution_trace: ExecutionTraceItem[]
  events: JobEvent[]
  round_timings: RoundTimingResponse[]
  feedback: FeedbackItem[]
}

export interface FeedbackItem {
  accepted: boolean
  rating?: number | null
  feedback_text?: string | null
  manual_adjustments: Record<string, unknown>
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
  focus_affinity?: string[]
  supports_mask?: boolean
  requires_mask?: boolean
  supports_whole_image?: boolean
  recommended_mask_prompt?: string | null
  recommended_mask_prompts?: string[]
  selection_guidance?: string
  conflict_tools?: string[]
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

export interface ToolLabMaskRequest {
  input_asset_id: string
  prompt: string
  provider?: 'auto' | 'aliyun' | 'fal_sam3'
}

export interface ToolLabMaskResponse {
  mask_asset: AssetResponse
  preview_asset?: AssetResponse | null
  provider: string
  requested_provider?: string | null
  prompt: string
  effective_prompt?: string | null
  fallback_used: boolean
  attempt_strategy?: string | null
  attempt_index?: number | null
  target_label?: string | null
  revert_mask?: boolean | null
}

export interface ToolLabStepRequest {
  tool_name: string
  params: Record<string, unknown>
  mask_asset_id?: string | null
}

export interface ToolLabStepResultResponse {
  index: number
  tool_name: string
  ok: boolean
  input_asset: AssetResponse
  output_asset?: AssetResponse | null
  mask_asset?: AssetResponse | null
  applied_params: Record<string, unknown>
  warnings: string[]
  artifacts: Record<string, unknown>
  fallback_used: boolean
  error?: string | null
}

export interface ToolLabRunRequest {
  input_asset_id: string
  steps: ToolLabStepRequest[]
}

export interface ToolLabRunResponse {
  input_asset: AssetResponse
  final_output_asset: AssetResponse
  steps: ToolLabStepResultResponse[]
}

export interface SseEventPayload extends JobEvent {}
