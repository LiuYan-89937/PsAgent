<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import {
  generateToolLabMask,
  listTools,
  runToolLab,
  uploadAssets,
} from '@/lib/api'
import type {
  AssetResponse,
  ToolCatalogItem,
  ToolLabMaskResponse,
  ToolLabRunResponse,
  ToolLabStepRequest,
} from '@/types/api'

type ProviderName = 'auto' | 'aliyun' | 'fal_sam3'

type ToolLabMaskItem = ToolLabMaskResponse & {
  id: string
  label: string
}

type StepDraft = {
  id: string
  toolName: string
  params: Record<string, unknown>
  maskAssetId: string | null
}

const toolCatalog = ref<ToolCatalogItem[]>([])
const catalogLoading = ref(false)
const catalogError = ref('')

const inputAsset = ref<AssetResponse | null>(null)
const inputError = ref('')
const inputUploading = ref(false)

const masks = ref<ToolLabMaskItem[]>([])
const maskPrompt = ref('person')
const maskProvider = ref<ProviderName>('fal_sam3')
const maskLoading = ref(false)
const maskError = ref('')
const maskUploading = ref(false)

const steps = ref<StepDraft[]>([])
const addToolName = ref('')
const runLoading = ref(false)
const runError = ref('')
const runResult = ref<ToolLabRunResponse | null>(null)
const selectedCompareIndex = ref(-1)

function randomId(prefix: string): string {
  return `${prefix}_${Math.random().toString(36).slice(2, 10)}`
}

function nonNullSchema(spec: Record<string, unknown> | undefined): Record<string, unknown> {
  if (!spec || typeof spec !== 'object') return {}
  const variants = Array.isArray(spec.anyOf) ? spec.anyOf : [spec]
  const picked = variants.find((item) => item && typeof item === 'object' && (item as Record<string, unknown>).type !== 'null')
  return (picked as Record<string, unknown>) || spec
}

function toolByName(name: string): ToolCatalogItem | undefined {
  return toolCatalog.value.find((item) => item.name === name)
}

function editableParamsForTool(toolName: string): Array<{ name: string; spec: Record<string, unknown> }> {
  const tool = toolByName(toolName)
  const properties = (tool?.planner_schema?.properties || {}) as Record<string, Record<string, unknown>>
  return Object.entries(properties)
    .filter(([name]) => !name.startsWith('mask_') && name !== 'image_path' && name !== 'mask_path')
    .map(([name, spec]) => ({ name, spec }))
}

function fieldType(spec: Record<string, unknown>): 'number' | 'integer' | 'boolean' | 'string' {
  const normalized = nonNullSchema(spec)
  const type = normalized.type
  if (type === 'integer') return 'integer'
  if (type === 'number') return 'number'
  if (type === 'boolean') return 'boolean'
  return 'string'
}

function fieldStep(spec: Record<string, unknown>): string {
  const normalized = nonNullSchema(spec)
  if (normalized.type === 'integer') return '1'
  const minimum = typeof normalized.minimum === 'number' ? normalized.minimum : undefined
  const maximum = typeof normalized.maximum === 'number' ? normalized.maximum : undefined
  if (minimum !== undefined && maximum !== undefined && Math.abs(maximum - minimum) <= 1) {
    return '0.01'
  }
  return '0.1'
}

function fieldEnum(spec: Record<string, unknown>): string[] {
  const normalized = nonNullSchema(spec)
  return Array.isArray(normalized.enum) ? normalized.enum.map(String) : []
}

function fieldMinimum(spec: Record<string, unknown>): number | undefined {
  const value = nonNullSchema(spec).minimum
  return typeof value === 'number' ? value : undefined
}

function fieldMaximum(spec: Record<string, unknown>): number | undefined {
  const value = nonNullSchema(spec).maximum
  return typeof value === 'number' ? value : undefined
}

function hasMaskSupport(toolName: string): boolean {
  return Boolean(toolByName(toolName)?.supports_mask)
}

function requiresMask(toolName: string): boolean {
  return Boolean(toolByName(toolName)?.requires_mask)
}

function recommendedMaskPrompt(toolName: string): string | null {
  const value = toolByName(toolName)?.recommended_mask_prompt
  return typeof value === 'string' && value.trim() ? value : null
}

function cloneDefaultParams(toolName: string): Record<string, unknown> {
  const defaults = { ...(toolByName(toolName)?.default_params || {}) }
  for (const key of Object.keys(defaults)) {
    if (key.startsWith('mask_')) delete defaults[key]
  }
  return defaults
}

function moveStep(index: number, direction: -1 | 1): void {
  const nextIndex = index + direction
  if (nextIndex < 0 || nextIndex >= steps.value.length) return
  const draft = [...steps.value]
  const [moved] = draft.splice(index, 1)
  draft.splice(nextIndex, 0, moved)
  steps.value = draft
}

function addStepFromSelectedTool(): void {
  if (!addToolName.value) return
  steps.value.push({
    id: randomId('step'),
    toolName: addToolName.value,
    params: cloneDefaultParams(addToolName.value),
    maskAssetId: null,
  })
  addToolName.value = ''
}

function removeStep(stepId: string): void {
  steps.value = steps.value.filter((item) => item.id !== stepId)
}

function updateStepTool(step: StepDraft, toolName: string): void {
  step.toolName = toolName
  step.params = cloneDefaultParams(toolName)
  if (!hasMaskSupport(toolName)) {
    step.maskAssetId = null
  }
}

function updateParam(step: StepDraft, paramName: string, rawValue: string | boolean, spec: Record<string, unknown>): void {
  const type = fieldType(spec)
  if (type === 'boolean') {
    step.params[paramName] = Boolean(rawValue)
    return
  }
  if (typeof rawValue !== 'string') return
  if (rawValue === '') {
    delete step.params[paramName]
    return
  }
  if (type === 'integer') {
    step.params[paramName] = Number.parseInt(rawValue, 10)
    return
  }
  if (type === 'number') {
    step.params[paramName] = Number.parseFloat(rawValue)
    return
  }
  step.params[paramName] = rawValue
}

function paramInputValue(step: StepDraft, paramName: string): string {
  const value = step.params[paramName]
  return value === undefined || value === null ? '' : String(value)
}

function paramBooleanValue(step: StepDraft, paramName: string): boolean {
  return Boolean(step.params[paramName])
}

function handleToolSelect(step: StepDraft, event: Event): void {
  const target = event.target as HTMLSelectElement | null
  if (!target) return
  updateStepTool(step, target.value)
}

function handleParamTextInput(step: StepDraft, paramName: string, spec: Record<string, unknown>, event: Event): void {
  const target = event.target as HTMLInputElement | HTMLSelectElement | null
  if (!target) return
  updateParam(step, paramName, target.value, spec)
}

function handleParamCheckboxInput(step: StepDraft, paramName: string, spec: Record<string, unknown>, event: Event): void {
  const target = event.target as HTMLInputElement | null
  if (!target) return
  updateParam(step, paramName, target.checked, spec)
}

async function handleMainImageUpload(event: Event): Promise<void> {
  const target = event.target as HTMLInputElement
  const files = target.files
  if (!files || files.length === 0) return
  inputUploading.value = true
  inputError.value = ''
  runResult.value = null
  masks.value = []
  steps.value = []
  try {
    const uploaded = await uploadAssets([files[0]])
    inputAsset.value = uploaded.items[0] || null
    if (!inputAsset.value) {
      inputError.value = '上传后没有拿到图片资源。'
    }
  } catch (error) {
    inputError.value = error instanceof Error ? error.message : '主图上传失败。'
  } finally {
    inputUploading.value = false
    target.value = ''
  }
}

async function handleMaskUpload(event: Event): Promise<void> {
  const target = event.target as HTMLInputElement
  const files = target.files
  if (!files || files.length === 0) return
  maskUploading.value = true
  maskError.value = ''
  try {
    const uploaded = await uploadAssets([files[0]])
    const asset = uploaded.items[0]
    if (asset) {
      masks.value.push({
        id: asset.asset_id,
        label: `上传遮罩: ${asset.filename}`,
        mask_asset: asset,
        preview_asset: asset,
        provider: 'manual_upload',
        requested_provider: null,
        prompt: asset.filename,
        effective_prompt: null,
        fallback_used: false,
        attempt_strategy: null,
        attempt_index: null,
        target_label: null,
        revert_mask: null,
      })
    }
  } catch (error) {
    maskError.value = error instanceof Error ? error.message : '遮罩上传失败。'
  } finally {
    maskUploading.value = false
    target.value = ''
  }
}

async function handleGenerateMask(): Promise<void> {
  if (!inputAsset.value || !maskPrompt.value.trim()) return
  maskLoading.value = true
  maskError.value = ''
  try {
    const generated = await generateToolLabMask({
      input_asset_id: inputAsset.value.asset_id,
      prompt: maskPrompt.value.trim(),
      provider: maskProvider.value,
    })
    masks.value.push({
      ...generated,
      id: generated.mask_asset.asset_id,
      label: `${generated.prompt} · ${generated.provider}`,
    })
  } catch (error) {
    maskError.value = error instanceof Error ? error.message : '调用遮罩生成失败。'
  } finally {
    maskLoading.value = false
  }
}

async function handleRunPipeline(): Promise<void> {
  if (!inputAsset.value || steps.value.length === 0) return
  runLoading.value = true
  runError.value = ''
  try {
    const payload: ToolLabStepRequest[] = steps.value.map((step) => ({
      tool_name: step.toolName,
      params: { ...step.params },
      mask_asset_id: step.maskAssetId || null,
    }))
    runResult.value = await runToolLab({
      input_asset_id: inputAsset.value.asset_id,
      steps: payload,
    })
    selectedCompareIndex.value = runResult.value.steps.length > 0 ? runResult.value.steps.length - 1 : -1
  } catch (error) {
    runError.value = error instanceof Error ? error.message : '工具链执行失败。'
  } finally {
    runLoading.value = false
  }
}

const groupedToolCatalog = computed(() => {
  const groups = new Map<string, ToolCatalogItem[]>()
  for (const item of toolCatalog.value) {
    const family = item.family || 'other'
    if (!groups.has(family)) groups.set(family, [])
    groups.get(family)!.push(item)
  }
  return Array.from(groups.entries())
})

const activeComparison = computed(() => {
  if (!inputAsset.value) return null
  if (!runResult.value || runResult.value.steps.length === 0) {
    return {
      title: '原图',
      beforeUrl: inputAsset.value.content_url,
      afterUrl: inputAsset.value.content_url,
      detail: '还没有运行工具链。',
    }
  }
  const stepsList = runResult.value.steps
  const step = stepsList[selectedCompareIndex.value] || stepsList[stepsList.length - 1]
  return {
    title: `${step.index + 1}. ${step.tool_name}`,
    beforeUrl: step.input_asset.content_url,
    afterUrl: step.output_asset?.content_url || runResult.value.final_output_asset.content_url,
    detail: step.ok ? '查看当前步骤的前后变化。' : (step.error || '当前步骤失败。'),
  }
})

onMounted(async () => {
  catalogLoading.value = true
  try {
    const catalog = await listTools()
    toolCatalog.value = catalog.items
  } catch (error) {
    catalogError.value = error instanceof Error ? error.message : '工具目录加载失败。'
  } finally {
    catalogLoading.value = false
  }
})
</script>

<template>
  <section class="tool-lab">
    <div class="tool-lab-grid">
      <aside class="tool-lab-sidebar">
        <section class="glass-panel tool-lab-panel">
          <div class="panel-head">
            <h2>工具实验室</h2>
            <p>自由排工具顺序、调整参数、接入遮罩，然后直接观察每一步的前后差异。</p>
          </div>

          <div class="panel-block">
            <label class="panel-label">主图片</label>
            <input class="system-file-input" type="file" accept="image/*" @change="handleMainImageUpload" />
            <p v-if="inputUploading" class="panel-hint">正在上传主图片...</p>
            <p v-if="inputError" class="panel-error">{{ inputError }}</p>
            <img v-if="inputAsset" :src="inputAsset.content_url" alt="Tool lab input" class="tool-lab-preview" />
          </div>

          <div class="panel-block">
            <div class="panel-head-inline">
              <div>
                <label class="panel-label">遮罩</label>
                <p class="panel-hint">可上传自己的 mask，也可直接调用 SAM 生成。</p>
              </div>
            </div>
            <div class="mask-form">
              <input v-model="maskPrompt" class="input-base" type="text" placeholder="例如 person / face / hair" />
              <select v-model="maskProvider" class="input-base">
                <option value="fal_sam3">fal_sam3</option>
                <option value="auto">auto</option>
                <option value="aliyun">aliyun</option>
              </select>
            </div>
            <div class="mask-actions">
              <button class="btn-primary" :disabled="!inputAsset || maskLoading" @click="handleGenerateMask">
                {{ maskLoading ? '生成中...' : '调用 SAM 生成遮罩' }}
              </button>
              <input class="system-file-input" type="file" accept="image/*" @change="handleMaskUpload" />
            </div>
            <p v-if="maskUploading" class="panel-hint">正在上传遮罩...</p>
            <p v-if="maskError" class="panel-error">{{ maskError }}</p>
            <div v-if="masks.length" class="mask-list">
              <article v-for="mask in masks" :key="mask.id" class="mask-card">
                <img :src="(mask.preview_asset || mask.mask_asset).content_url" :alt="mask.label" class="mask-thumb" />
                <div class="mask-meta">
                  <strong>{{ mask.label }}</strong>
                  <span>{{ mask.mask_asset.filename }}</span>
                </div>
              </article>
            </div>
          </div>

          <div class="panel-block">
            <div class="panel-head-inline">
              <div>
                <label class="panel-label">工具链</label>
                <p class="panel-hint">顺序就是实际执行顺序，上一步输出会成为下一步输入。</p>
              </div>
            </div>
            <div class="add-tool-row">
              <select v-model="addToolName" class="input-base">
                <option value="">选择工具</option>
                <optgroup
                  v-for="[family, items] in groupedToolCatalog"
                  :key="family"
                  :label="family"
                >
                  <option v-for="tool in items" :key="tool.name" :value="tool.name">
                    {{ tool.label || tool.name }}
                  </option>
                </optgroup>
              </select>
              <button class="btn-primary" :disabled="!addToolName" @click="addStepFromSelectedTool">添加步骤</button>
            </div>
            <p v-if="catalogLoading" class="panel-hint">正在加载工具目录...</p>
            <p v-if="catalogError" class="panel-error">{{ catalogError }}</p>

            <div v-if="steps.length" class="step-list">
              <article v-for="(step, index) in steps" :key="step.id" class="step-card">
                <div class="step-head">
                  <strong>Step {{ index + 1 }}</strong>
                  <div class="step-actions">
                    <button class="ghost-btn" :disabled="index === 0" @click="moveStep(index, -1)">上移</button>
                    <button class="ghost-btn" :disabled="index === steps.length - 1" @click="moveStep(index, 1)">下移</button>
                    <button class="ghost-btn danger" @click="removeStep(step.id)">删除</button>
                  </div>
                </div>

                <div class="step-body">
                  <label class="field-label">
                    工具
                    <select class="input-base" :value="step.toolName" @change="handleToolSelect(step, $event)">
                      <option v-for="tool in toolCatalog" :key="tool.name" :value="tool.name">
                        {{ tool.label || tool.name }}
                      </option>
                    </select>
                  </label>

                  <label class="field-label">
                    遮罩
                    <select class="input-base" :disabled="!hasMaskSupport(step.toolName)" v-model="step.maskAssetId">
                      <option v-if="!requiresMask(step.toolName)" :value="null">不使用遮罩</option>
                      <option v-for="mask in masks" :key="mask.id" :value="mask.mask_asset.asset_id">
                        {{ mask.label }}
                      </option>
                    </select>
                    <small v-if="requiresMask(step.toolName)" class="field-help">
                      这个工具必须使用局部遮罩。
                      <template v-if="recommendedMaskPrompt(step.toolName)">
                        推荐先生成 `{{ recommendedMaskPrompt(step.toolName) }}` 遮罩。
                      </template>
                    </small>
                  </label>

                  <div class="params-grid">
                    <label
                      v-for="field in editableParamsForTool(step.toolName)"
                      :key="`${step.id}_${field.name}`"
                      class="field-label"
                    >
                      <span>{{ field.name }}</span>
                      <template v-if="fieldEnum(field.spec).length > 0">
                        <select
                          class="input-base"
                          :value="paramInputValue(step, field.name)"
                          @change="handleParamTextInput(step, field.name, field.spec, $event)"
                        >
                          <option value="">默认</option>
                          <option v-for="option in fieldEnum(field.spec)" :key="option" :value="option">{{ option }}</option>
                        </select>
                      </template>
                      <template v-else-if="fieldType(field.spec) === 'boolean'">
                        <label class="bool-field">
                          <input
                            type="checkbox"
                            :checked="paramBooleanValue(step, field.name)"
                            @change="handleParamCheckboxInput(step, field.name, field.spec, $event)"
                          />
                          <span>{{ String(field.spec.description || '布尔开关') }}</span>
                        </label>
                      </template>
                      <template v-else>
                        <input
                          class="input-base"
                          :type="fieldType(field.spec) === 'string' ? 'text' : 'number'"
                          :step="fieldStep(field.spec)"
                          :min="fieldMinimum(field.spec)"
                          :max="fieldMaximum(field.spec)"
                          :placeholder="String(field.spec.description || '')"
                          :value="paramInputValue(step, field.name)"
                          @input="handleParamTextInput(step, field.name, field.spec, $event)"
                        />
                      </template>
                      <small v-if="field.spec.description" class="field-help">{{ field.spec.description }}</small>
                    </label>
                  </div>
                </div>
              </article>
            </div>
          </div>

          <div class="panel-block panel-run">
            <button class="btn-primary" :disabled="!inputAsset || steps.length === 0 || runLoading" @click="handleRunPipeline">
              {{ runLoading ? '正在执行工具链...' : '运行工具链' }}
            </button>
            <p v-if="runError" class="panel-error">{{ runError }}</p>
          </div>
        </section>
      </aside>

      <section class="tool-lab-main">
        <div class="glass-panel tool-lab-panel compare-panel">
          <div class="panel-head-inline">
            <div>
              <h3>前后对比</h3>
              <p>{{ activeComparison?.detail }}</p>
            </div>
            <div class="compare-label">{{ activeComparison?.title }}</div>
          </div>

          <div v-if="activeComparison" class="compare-slot">
            <div class="compare-pane">
              <span class="compare-badge">Before</span>
              <img :src="activeComparison.beforeUrl" alt="Before" class="compare-image" />
            </div>
            <div class="compare-pane">
              <span class="compare-badge compare-badge-highlight">After</span>
              <img :src="activeComparison.afterUrl" alt="After" class="compare-image" />
            </div>
          </div>

          <div v-if="runResult" class="result-switcher">
            <button
              v-for="step in runResult.steps"
              :key="step.index"
              class="result-chip"
              :class="{ active: selectedCompareIndex === step.index }"
              @click="selectedCompareIndex = step.index"
            >
              {{ step.index + 1 }} · {{ step.tool_name }}
            </button>
          </div>
        </div>

        <div v-if="runResult" class="glass-panel tool-lab-panel results-panel">
          <div class="panel-head-inline">
            <div>
              <h3>执行结果</h3>
              <p>点击步骤卡片可以在上方查看对应的前后对比。</p>
            </div>
            <a class="result-link" :href="runResult.final_output_asset.content_url" target="_blank" rel="noreferrer">
              打开最终输出
            </a>
          </div>

          <div class="result-grid">
            <article
              v-for="step in runResult.steps"
              :key="`result_${step.index}`"
              class="result-card"
              :class="{ active: selectedCompareIndex === step.index }"
              @click="selectedCompareIndex = step.index"
            >
              <div class="result-card-head">
                <strong>{{ step.index + 1 }}. {{ step.tool_name }}</strong>
                <span :class="step.ok ? 'status-ok' : 'status-fail'">{{ step.ok ? '成功' : '失败' }}</span>
              </div>
              <div class="result-thumbs">
                <img :src="step.input_asset.content_url" alt="Step input" class="result-thumb" />
                <img v-if="step.output_asset" :src="step.output_asset.content_url" alt="Step output" class="result-thumb" />
              </div>
              <p v-if="step.mask_asset" class="result-mask-note">遮罩：{{ step.mask_asset.filename }}</p>
              <p v-if="step.error" class="panel-error">{{ step.error }}</p>
            </article>
          </div>
        </div>
      </section>
    </div>
  </section>
</template>

<style scoped>
.tool-lab {
  width: 100%;
}

.tool-lab-grid {
  display: grid;
  grid-template-columns: 420px minmax(0, 1fr);
  gap: 20px;
  align-items: start;
}

.tool-lab-sidebar,
.tool-lab-main {
  min-width: 0;
}

.tool-lab-panel {
  padding: 20px;
}

.tool-lab-main {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.panel-head h2,
.panel-head-inline h3 {
  margin: 0;
}

.panel-head p,
.panel-head-inline p,
.panel-hint {
  margin: 6px 0 0;
  color: var(--text-muted);
  font-size: 0.92rem;
}

.panel-head-inline {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
}

.compare-label {
  padding: 6px 12px;
  border-radius: 999px;
  border: 1px solid var(--border-glass);
  color: var(--text-muted);
  white-space: nowrap;
}

.panel-block + .panel-block {
  margin-top: 24px;
  padding-top: 24px;
  border-top: 1px solid var(--border-glass);
}

.panel-label {
  display: block;
  margin-bottom: 10px;
  color: var(--text-main);
  font-weight: 600;
}

.tool-lab-preview {
  width: 100%;
  max-height: 240px;
  margin-top: 12px;
  object-fit: contain;
  border-radius: 14px;
  border: 1px solid var(--border-glass);
  background: rgba(0, 0, 0, 0.25);
}

.system-file-input {
  display: block;
  width: 100%;
  color: var(--text-main);
}

.mask-form,
.add-tool-row,
.mask-actions {
  display: grid;
  grid-template-columns: 1fr 140px;
  gap: 10px;
}

.mask-actions {
  grid-template-columns: 1fr;
  margin-top: 12px;
}

.mask-list {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
  margin-top: 14px;
}

.mask-card {
  display: flex;
  gap: 10px;
  padding: 10px;
  border: 1px solid var(--border-glass);
  border-radius: 14px;
  background: rgba(255, 255, 255, 0.03);
}

.mask-thumb {
  width: 72px;
  height: 72px;
  object-fit: cover;
  border-radius: 10px;
  border: 1px solid var(--border-glass);
}

.mask-meta {
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 0;
}

.mask-meta span {
  color: var(--text-muted);
  font-size: 0.82rem;
  word-break: break-all;
}

.step-list {
  display: flex;
  flex-direction: column;
  gap: 14px;
  margin-top: 14px;
}

.step-card {
  padding: 14px;
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.03);
  border: 1px solid var(--border-glass);
}

.step-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 14px;
}

.step-actions {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.ghost-btn {
  border: 1px solid var(--border-glass);
  background: rgba(255, 255, 255, 0.04);
  color: var(--text-main);
  border-radius: 999px;
  padding: 6px 10px;
  cursor: pointer;
}

.ghost-btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.ghost-btn.danger {
  color: #fca5a5;
}

.step-body {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.field-label {
  display: flex;
  flex-direction: column;
  gap: 6px;
  min-width: 0;
}

.params-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
}

.field-help {
  color: var(--text-muted);
  font-size: 0.78rem;
  line-height: 1.45;
}

.bool-field {
  display: flex;
  align-items: center;
  gap: 8px;
  min-height: 46px;
  padding: 0 2px;
  color: var(--text-muted);
}

.panel-run {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.panel-error {
  margin: 10px 0 0;
  color: #fca5a5;
  font-size: 0.9rem;
}

.compare-panel,
.results-panel {
  overflow: hidden;
}

.compare-slot {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px;
  margin-top: 16px;
}

.compare-pane {
  position: relative;
  min-width: 0;
  aspect-ratio: 2 / 3;
  border-radius: 18px;
  overflow: hidden;
  border: 1px solid var(--border-glass);
  background: rgba(0, 0, 0, 0.28);
}

.compare-badge {
  position: absolute;
  top: 14px;
  left: 14px;
  z-index: 2;
  padding: 6px 12px;
  border-radius: 999px;
  background: rgba(0, 0, 0, 0.62);
  color: #fff;
  font-size: 0.82rem;
  letter-spacing: 0.02em;
}

.compare-badge-highlight {
  background: rgba(99, 102, 241, 0.8);
}

.compare-image {
  width: 100%;
  height: 100%;
  object-fit: contain;
  display: block;
}

.result-switcher {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 16px;
}

.result-chip {
  border: 1px solid var(--border-glass);
  background: rgba(255, 255, 255, 0.04);
  color: var(--text-main);
  border-radius: 999px;
  padding: 8px 12px;
  cursor: pointer;
}

.result-chip.active {
  border-color: rgba(99, 102, 241, 0.4);
  background: rgba(99, 102, 241, 0.16);
}

.result-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
  margin-top: 16px;
}

.result-card {
  padding: 14px;
  border-radius: 16px;
  border: 1px solid var(--border-glass);
  background: rgba(255, 255, 255, 0.03);
  cursor: pointer;
}

.result-card.active {
  border-color: rgba(99, 102, 241, 0.5);
  background: rgba(99, 102, 241, 0.1);
}

.result-card-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 10px;
}

.result-thumbs {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 8px;
}

.result-thumb {
  width: 100%;
  aspect-ratio: 1;
  object-fit: cover;
  border-radius: 12px;
  border: 1px solid var(--border-glass);
}

.result-mask-note {
  margin: 10px 0 0;
  color: var(--text-muted);
  font-size: 0.84rem;
}

.result-link {
  color: var(--accent-primary);
  text-decoration: none;
}

.status-ok {
  color: #6ee7b7;
}

.status-fail {
  color: #fca5a5;
}

@media (max-width: 1180px) {
  .tool-lab-grid {
    grid-template-columns: 1fr;
  }

  .params-grid,
  .result-grid,
  .mask-list {
    grid-template-columns: 1fr;
  }

  .mask-form,
  .add-tool-row {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 900px) {
  .compare-slot {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 680px) {
  .panel-head-inline {
    flex-direction: column;
  }

  .step-head {
    flex-direction: column;
    align-items: flex-start;
  }

  .result-thumbs {
    grid-template-columns: 1fr;
  }
}
</style>
