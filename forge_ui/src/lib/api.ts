// REST + SSE client for the ForgeAI GUI backend.

const BASE = ''

// Per-launch auth token: the launcher opens `/?token=…`; we lift it into
// sessionStorage (per-tab) and strip it from the visible URL. The server
// never serves it back over HTTP.
function initToken(): string {
  const q = new URLSearchParams(location.search).get('token')
  if (q) {
    sessionStorage.setItem('forge_gui_token', q)
    const url = new URL(location.href)
    url.searchParams.delete('token')
    history.replaceState(null, '', url.toString())
    return q
  }
  return sessionStorage.getItem('forge_gui_token') ?? ''
}

export const authToken = initToken()

function authHeaders(extra?: Record<string, string>): Record<string, string> {
  return { 'X-Forge-Token': authToken, ...(extra ?? {}) }
}

async function req<T>(method: string, path: string, body?: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method,
    headers: authHeaders(body !== undefined ? { 'Content-Type': 'application/json' } : undefined),
    body: body !== undefined ? JSON.stringify(body) : undefined,
  })
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`
    try {
      const j = await res.json()
      detail = j.detail || j.error || detail
    } catch { /* keep status text */ }
    throw new Error(detail)
  }
  return res.json() as Promise<T>
}

export const api = {
  get: <T>(path: string) => req<T>('GET', path),
  post: <T>(path: string, body?: unknown) => req<T>('POST', path, body ?? {}),
  del: <T>(path: string) => req<T>('DELETE', path),
}

export interface SseEvent {
  type: string
  data: unknown
}

/**
 * POST with an SSE response. Calls onEvent for each `data: {json}` frame.
 * Returns when the stream closes; abort via the AbortSignal.
 */
export async function ssePost(
  path: string,
  body: unknown,
  onEvent: (evt: SseEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: authHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify(body),
    signal,
  })
  if (!res.ok || !res.body) {
    throw new Error(`${res.status} ${res.statusText}`)
  }
  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buf = ''
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buf += decoder.decode(value, { stream: true })
    let idx: number
    while ((idx = buf.indexOf('\n\n')) >= 0) {
      const frame = buf.slice(0, idx)
      buf = buf.slice(idx + 2)
      for (const line of frame.split('\n')) {
        if (line.startsWith('data: ')) {
          try {
            onEvent(JSON.parse(line.slice(6)) as SseEvent)
          } catch { /* malformed frame — skip */ }
        }
      }
    }
  }
}
