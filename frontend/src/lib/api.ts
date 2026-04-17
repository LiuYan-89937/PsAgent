import type {
  EditRequest,
  EditResponse,
  FeedbackRequest,
  JobDetailResponse,
  PackageCatalogResponse,
  ResumeReviewRequest,
  ResumeReviewResponse,
  UploadAssetsResponse,
  SseEventPayload,
} from '@/types/api'

export const apiBaseUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

function extractErrorMessage(payload: unknown, fallback: string): string {
  if (typeof payload === 'string') return payload
  if (payload && typeof payload === 'object') {
    const detail = (payload as Record<string, unknown>).detail
    if (typeof detail === 'string') return detail
    if (detail && typeof detail === 'object' && typeof (detail as Record<string, unknown>).message === 'string') {
      return String((detail as Record<string, unknown>).message)
    }
    if (typeof (payload as Record<string, unknown>).message === 'string') {
      return String((payload as Record<string, unknown>).message)
    }
  }
  return fallback
}

async function buildHttpError(response: Response): Promise<Error> {
  let payload: unknown = null
  try {
    payload = await response.json()
  } catch {
    try {
      payload = await response.text()
    } catch {
      payload = null
    }
  }
  return new Error(extractErrorMessage(payload, `HTTP ${response.status}`))
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${apiBaseUrl}${path}`, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers || {}),
    },
  })

  if (!response.ok) {
    throw await buildHttpError(response)
  }

  return (await response.json()) as T
}

export async function uploadAssets(files: File[]): Promise<UploadAssetsResponse> {
  const formData = new FormData()
  files.forEach((file) => {
    formData.append('files', file)
  })

  const response = await fetch(`${apiBaseUrl}/assets/upload`, {
    method: 'POST',
    body: formData,
  })
  if (!response.ok) {
    throw await buildHttpError(response)
  }
  return (await response.json()) as UploadAssetsResponse
}

export function listPackages(): Promise<PackageCatalogResponse> {
  return requestJson<PackageCatalogResponse>('/meta/packages')
}

export function submitEdit(payload: EditRequest): Promise<EditResponse> {
  return requestJson<EditResponse>('/edit', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export function getJob(jobId: string): Promise<JobDetailResponse> {
  return requestJson<JobDetailResponse>(`/jobs/${jobId}`)
}

export function submitFeedback(payload: FeedbackRequest): Promise<{ job_id: string; saved: boolean; feedback_count: number }> {
  return requestJson('/feedback', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export function resumeReview(payload: ResumeReviewRequest): Promise<ResumeReviewResponse> {
  return requestJson<ResumeReviewResponse>('/resume-review', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export async function streamEdit(
  payload: EditRequest,
  onEvent: (eventName: string, data: SseEventPayload) => void,
): Promise<void> {
  const response = await fetch(`${apiBaseUrl}/edit/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  })

  if (!response.ok || !response.body) {
    throw await buildHttpError(response)
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    const blocks = buffer.split('\n\n')
    buffer = blocks.pop() || ''

    for (const block of blocks) {
      const lines = block.split('\n')
      const eventLine = lines.find((line) => line.startsWith('event:'))
      const dataLine = lines.find((line) => line.startsWith('data:'))
      if (!eventLine || !dataLine) continue

      const eventName = eventLine.replace(/^event:\s*/, '').trim()
      const payloadText = dataLine.replace(/^data:\s*/, '').trim()
      const data = JSON.parse(payloadText) as SseEventPayload
      onEvent(eventName, data)
    }
  }
}
