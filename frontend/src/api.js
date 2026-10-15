import axios from 'axios'

const TOKEN_KEY = 'audi_token'

export const getToken = () => sessionStorage.getItem(TOKEN_KEY)
export const setToken = (t) => (t ? sessionStorage.setItem(TOKEN_KEY, t) : sessionStorage.removeItem(TOKEN_KEY))

const api = axios.create({ baseURL: '/api' })

api.interceptors.request.use((cfg) => {
  const t = getToken()
  if (t) cfg.headers.Authorization = `Bearer ${t}`
  return cfg
})

let onUnauthorized = () => {}
export const setUnauthorizedHandler = (fn) => { onUnauthorized = fn }
api.interceptors.response.use(
  (r) => r,
  (err) => {
    if (err.response?.status === 401 && !err.config.url.startsWith('/auth/token')) onUnauthorized()
    return Promise.reject(err)
  },
)

export const errorText = (err) => {
  const d = err?.response?.data?.detail
  if (Array.isArray(d)) return d.map((x) => `${x.loc?.slice(-1)[0]}: ${x.msg}`).join('; ')
  return d || err?.message || 'Request failed'
}

const data = (p) => p.then((r) => r.data)

// auth
export const login = (username, password) => {
  const form = new URLSearchParams({ username, password })
  return data(api.post('/auth/token', form))
}
export const me = () => data(api.get('/auth/me'))

// public
export const getHealth = () => data(api.get('/health'))

// sites, sessions, reports
export const getSites = () => data(api.get('/sites'))
export const getBriefing = (siteId) => data(api.get(`/sites/${siteId}/briefing`))
export const getSiteContext = (siteId) => data(api.get(`/sites/${siteId}/context`))
export const refreshSiteContext = (siteId) => data(api.post(`/admin/sites/${siteId}/context/refresh`))
export const setSiteContext = (siteId, body) => data(api.put(`/admin/sites/${siteId}/context`, body))
export const quickStart = (location) => data(api.post('/sessions/quick-start', location || {}))
export const startSession = (siteId, location) => data(api.post('/sessions/start', { site_id: siteId, ...(location || {}) }))
export const endSession = (id) => data(api.post(`/sessions/${id}/end`))
export const getSessions = (siteId) => data(api.get('/sessions', { params: { site_id: siteId || undefined } }))
export const getInterrupted = () => data(api.get('/sessions/interrupted'))
export const resumeSession = (id) => data(api.post(`/sessions/${id}/resume`))
export const getSession = (id) => data(api.get(`/sessions/${id}`))
export const getReport = (id) => data(api.get(`/sessions/${id}/report`, { params: { format: 'json' } }))
export const getReportFile = (id, format) =>
  data(api.get(`/sessions/${id}/report`, { params: { format }, responseType: 'blob' }))

// assets and memory
export const getSummary = (siteId) => data(api.get('/summary', { params: { site_id: siteId || undefined } }))
export const getAssets = (siteId) => data(api.get('/assets', { params: { site_id: siteId || undefined } }))
export const getAsset = (id) => data(api.get(`/assets/${id}`))
export const getAssetHistory = (id) => data(api.get(`/assets/${id}/history`))
export const imageUrl = (ref) => (ref ? `/api/images/${ref}?token=${encodeURIComponent(getToken() || '')}` : null)
export const analyzeImage = (file, assetType, sessionId, addToMemory) => {
  const form = new FormData()
  form.append('file', file)
  form.append('asset_type', assetType)
  if (sessionId) form.append('session_id', sessionId)
  form.append('add_to_memory', addToMemory ? 'true' : 'false')
  return data(api.post('/analyze', form))
}
export const searchMemory = (file, assetType, siteId, topK) => {
  const form = new FormData()
  form.append('file', file)
  form.append('asset_type', assetType)
  if (siteId) form.append('site_id', siteId)
  form.append('top_k', String(topK))
  return data(api.post('/memory/search', form))
}

// defects
export const getDefects = (params) => data(api.get('/defects', { params }))
export const getDefect = (id) => data(api.get(`/defects/${id}`))
export const updateDefect = (id, status, notes) => data(api.patch(`/defects/${id}`, { status, notes }))

// notifications
export const getNotifications = (unreadOnly = false) => data(api.get('/notifications', { params: { unread_only: unreadOnly } }))
export const markRead = (id) => data(api.post(`/notifications/${id}/read`))

// live
export const liveStart = (body) => data(api.post('/live/start', body))
export const liveStop = () => data(api.post('/live/stop'))
export const liveStatus = () => data(api.get('/live/status'))
export const liveCapture = (trackId) => data(api.post('/live/capture', { track_id: trackId ?? null }))
export const liveStreamUrl = () => `/api/live/stream?token=${encodeURIComponent(getToken() || '')}&t=${Date.now()}`

// admin
export const getConfig = () => data(api.get('/admin/config'))
export const putConfig = (changes) => data(api.put('/admin/config', { changes }))
export const getConfigVersions = () => data(api.get('/admin/config/versions'))
export const createSite = (body) => data(api.post('/admin/sites', body))
export const getUsers = () => data(api.get('/admin/users'))
export const createUser = (body) => data(api.post('/admin/users', body))
export const updateUser = (id, body) => data(api.patch(`/admin/users/${id}`, body))
export const setUserSites = (id, sites) => data(api.put(`/admin/users/${id}/sites`, { sites }))
export const getSync = () => data(api.get('/admin/sync'))
export const flushSync = () => data(api.post('/admin/sync/flush'))
export const getMetrics = () => data(api.get('/admin/metrics'))

// websockets
export const wsUrl = (path) => {
  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
  return `${proto}://${window.location.host}${path}?token=${encodeURIComponent(getToken() || '')}`
}

export default api
