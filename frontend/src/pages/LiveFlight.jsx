import { useCallback, useEffect, useRef, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { endSession, errorText, getSession, liveCapture, liveStart, liveStatus, liveStop, liveStreamUrl } from '../api'
import { Card, ErrorBox, fmt, StatusPill, when } from '../components/ui'
import { useAsync, useSocket } from '../hooks'
import { ALERTS, BROWSER_CAMERA, STREAM_RETRY_MS } from '../settings'

const ASSET_TYPES = ['solar_panel', 'building', 'utility_pole', 'machinery']

function useBeeper() {
  const ctx = useRef(null)
  const arm = () => {
    if (!ctx.current) ctx.current = new (window.AudioContext || window.webkitAudioContext)()
    if (ctx.current.state === 'suspended') ctx.current.resume()
  }
  const play = ({ freq, ms, repeats }) => {
    const ac = ctx.current
    if (!ac) return
    for (let i = 0; i < repeats; i += 1) {
      const t0 = ac.currentTime + (i * 2 * ms) / 1000
      const osc = ac.createOscillator()
      const gain = ac.createGain()
      osc.frequency.value = freq
      gain.gain.setValueAtTime(0.2, t0)
      gain.gain.exponentialRampToValueAtTime(0.001, t0 + ms / 1000)
      osc.connect(gain).connect(ac.destination)
      osc.start(t0)
      osc.stop(t0 + ms / 1000)
    }
  }
  return { arm, play }
}

function BrowserCamera({ active, send, buffered, onError }) {
  const video = useRef(null)
  const canvas = useRef(null)
  useEffect(() => {
    if (!active) return undefined
    let stream = null
    let timer = null
    navigator.mediaDevices
      .getUserMedia({ video: { width: BROWSER_CAMERA.width, height: BROWSER_CAMERA.height }, audio: false })
      .then((s) => {
        stream = s
        video.current.srcObject = s
        timer = setInterval(() => {
          const v = video.current
          if (!v || v.readyState < 2 || buffered() > BROWSER_CAMERA.maxBufferedBytes) return
          const c = canvas.current
          c.width = v.videoWidth
          c.height = v.videoHeight
          c.getContext('2d').drawImage(v, 0, 0)
          c.toBlob((b) => b && b.arrayBuffer().then(send), 'image/jpeg', BROWSER_CAMERA.jpegQuality)
        }, BROWSER_CAMERA.frameIntervalMs)
      })
      .catch((e) => onError(`Browser camera: ${e.message}`))
    return () => {
      clearInterval(timer)
      stream?.getTracks().forEach((t) => t.stop())
    }
  }, [active, send, buffered, onError])
  return (
    <>
      <video ref={video} autoPlay muted playsInline className="hidden" />
      <canvas ref={canvas} className="hidden" />
    </>
  )
}

export default function LiveFlight() {
  const { sessionId } = useParams()
  const nav = useNavigate()
  const session = useAsync(() => getSession(sessionId), [sessionId])
  const [form, setForm] = useState({ mode: 'camera', camera_source: '', detection_mode: 'auto', asset_type: 'solar_panel' })
  const [seen, setSeen] = useState({})
  const [status, setStatus] = useState(null)
  const [alerts, setAlerts] = useState([])
  const [banner, setBanner] = useState(null)
  const [error, setError] = useState(null)
  const [streamKey, setStreamKey] = useState(liveStreamUrl())
  const [ending, setEnding] = useState(false)
  const beeper = useBeeper()
  const bannerTimer = useRef(null)

  const running = !!status?.running && status.session_id === sessionId

  const refresh = useCallback(() => liveStatus().then(setStatus).catch((e) => setError(errorText(e))), [])
  useEffect(() => { refresh() }, [refresh])

  const onMessage = useCallback((msg) => {
    if (msg.type === 'status') setStatus((s) => ({ ...(s || {}), ...msg.status }))
    if (msg.type === 'source_finished') refresh()
    if (msg.type === 'error') setError(msg.error)
    if (msg.type === 'event' && msg.event.asset_id && ['new_asset', 'confirmed', 'change', 'duplicate'].includes(msg.event.type)) {
      const ev = msg.event
      // Memory panel: keep what memory held BEFORE this session (first event), update the current state.
      setSeen((m) => {
        const prev = m[ev.asset_id]
        return {
          ...m,
          [ev.asset_id]: {
            asset_id: ev.asset_id, asset_type: ev.asset_type, firstType: prev ? prev.firstType : ev.type,
            history: prev ? prev.history : ev.history, context: ev.context || prev?.context,
            label: ev.condition?.label, confidence: ev.condition?.confidence, severity: ev.severity_score,
            level: ev.severity_level, change: ev.change, ts: ev.timestamp, count: (prev?.count || 0) + 1,
          },
        }
      })
    }
    if (msg.type === 'event' && msg.alert) {
      const ev = msg.event
      const a = {
        key: `${ev.observation_id || ev.asset_id}-${ev.timestamp}`, level: msg.alert, asset_id: ev.asset_id,
        label: ev.condition?.label, severity: ev.severity_score, ts: ev.timestamp, latency: msg.alert_latency_ms,
        history: ev.history,
      }
      setAlerts((list) => [a, ...list].slice(0, ALERTS.maxShown))
      setBanner(a)
      clearTimeout(bannerTimer.current)
      bannerTimer.current = setTimeout(() => setBanner(null), ALERTS.bannerMs)
      beeper.play(msg.alert === 'critical' ? ALERTS.criticalTone : ALERTS.priorityTone)
    }
  }, [beeper, refresh])
  const sock = useSocket('/ws/live', onMessage, true)

  const start = async () => {
    beeper.arm()
    setError(null)
    const body = { session_id: sessionId, mode: form.mode }
    if (form.mode === 'camera' && form.camera_source.trim() !== '') {
      body.camera_source = /^\d+$/.test(form.camera_source.trim()) ? Number(form.camera_source.trim()) : form.camera_source.trim()
    }
    if (form.detection_mode === 'full_frame') {
      body.detection_mode = 'full_frame'
      body.asset_type = form.asset_type
    } else if (form.detection_mode === 'auto') {
      body.detection_mode = 'auto'
    }
    try {
      setStatus(await liveStart(body))
      setStreamKey(liveStreamUrl())
    } catch (e) { setError(errorText(e)) }
  }
  const stop = async () => {
    try { setStatus(await liveStop()) } catch (e) { setError(errorText(e)) }
  }
  const capture = async () => {
    try { await liveCapture(null) } catch (e) { setError(errorText(e)) }
  }
  const end = async () => {
    setEnding(true)
    try {
      await endSession(sessionId)
      nav(`/report/${sessionId}`)
    } catch (e) {
      setError(errorText(e))
      setEnding(false)
    }
  }

  const s = session.data
  const lat = status?.latency_ms || {}
  const critical = banner?.level === 'critical'
  return (
    <div className="page">
      <header className="page-head">
        <h2>Live flight <span className="mono small">{sessionId}</span></h2>
        <div className="row gap">
          {s && <StatusPill status={s.status} />}
          {s && <StatusPill status={s.mode} />}
          <button className="btn danger" onClick={end} disabled={ending || s?.status !== 'active'}>{ending ? 'Ending & syncing…' : 'End session'}</button>
        </div>
      </header>
      <ErrorBox error={error || session.error} />
      {s && s.status !== 'active' && <div className="muted">This session has ended — <Link to={`/report/${sessionId}`}>open its report</Link>.</div>}

      <div className="live-grid">
        <div className={`live-view ${critical ? 'critical' : ''}`}>
          {banner && (
            <div className={`edge-banner ${banner.level}`}>
              {banner.level === 'critical' ? 'CRITICAL' : 'PRIORITY'} · {banner.asset_id} · {banner.label} · S {fmt(banner.severity)}
            </div>
          )}
          {running ? (
            <img key={streamKey} src={streamKey} alt="Annotated live feed" className="stream"
              onError={() => setTimeout(() => setStreamKey(liveStreamUrl()), STREAM_RETRY_MS)} />
          ) : (
            <div className="stream-off muted">Live input stopped</div>
          )}
          <BrowserCamera active={running && form.mode === 'browser'} send={sock.send} buffered={sock.bufferedAmount} onError={setError} />
        </div>

        <div className="live-side">
          <Card title="Input">
            {!running ? (
              <div className="stack">
                <label className="field"><span>Source</span>
                  <select value={form.mode} onChange={(e) => setForm({ ...form, mode: e.target.value })}>
                    <option value="camera">Camera on the API machine (USB webcam / RTSP / video file)</option>
                    <option value="browser">This browser's camera</option>
                  </select>
                </label>
                {form.mode === 'camera' && (
                  <label className="field"><span>Camera index, device name, URL or file</span>
                    <input value={form.camera_source} onChange={(e) => setForm({ ...form, camera_source: e.target.value })} aria-describedby="src-hint" />
                    <small id="src-hint" className="muted">Empty = <code>camera.source</code> from config.yaml (see scripts/camera_probe.py)</small>
                  </label>
                )}
                <label className="field"><span>Detection</span>
                  <select value={form.detection_mode} onChange={(e) => setForm({ ...form, detection_mode: e.target.value })}>
                    <option value="auto">Automatic (detects objects, or recognises the asset type of a close-up)</option>
                    <option value="">Configured detector (YOLO-World prompts)</option>
                    <option value="full_frame">Whole frame is one asset (close-up desk demo)</option>
                  </select>
                </label>
                {form.detection_mode === 'full_frame' && (
                  <label className="field"><span>Asset type</span>
                    <select value={form.asset_type} onChange={(e) => setForm({ ...form, asset_type: e.target.value })}>
                      {ASSET_TYPES.map((t) => <option key={t} value={t}>{t}</option>)}
                    </select>
                  </label>
                )}
                <button className="btn primary" onClick={start} disabled={s?.status !== 'active'}>Start live input</button>
              </div>
            ) : (
              <div className="row gap wrap">
                <button className="btn" onClick={capture} title="Force a full observation of the largest visible asset">Capture this asset</button>
                <button className="btn ghost" onClick={stop}>Stop input</button>
              </div>
            )}
          </Card>

          <Card title="Status">
            <dl className="kv">
              <dt>Camera</dt><dd><span className={`dot ${status?.camera_connected ? 'ok' : 'crit'}`} /> {status?.camera_connected ? 'connected' : 'not connected'}{status?.camera?.reconnects ? ` · ${status.camera.reconnects} reconnects` : ''}</dd>
              <dt>Capture fps</dt><dd>{fmt(status?.capture_fps, 1)}</dd>
              <dt>Processed fps</dt><dd>{fmt(status?.processed_fps, 1)}</dd>
              <dt>Alert latency p50 / p95</dt><dd>{fmt(lat.alert?.p50, 0)} / {fmt(lat.alert?.p95, 0)} ms</dd>
              <dt>Frame latency p50 / p95</dt><dd>{fmt(lat.capture_to_result?.p50, 0)} / {fmt(lat.capture_to_result?.p95, 0)} ms</dd>
              <dt>Frames seen / processed</dt><dd>{status?.counters?.seen ?? 0} / {status?.counters?.processed ?? 0}</dd>
              <dt>Skipped (duplicate / frame-skip)</dt><dd>{status?.counters?.skipped_duplicate ?? 0} / {status?.counters?.skipped_frame_skip ?? 0}</dd>
              <dt>Events socket</dt><dd><span className={`dot ${sock.connected ? 'ok' : 'warn'}`} /> {sock.connected ? 'connected' : 'reconnecting'}</dd>
            </dl>
          </Card>

          <Card title={`Memory: assets seen this session (${Object.keys(seen).length})`}>
            {Object.keys(seen).length === 0 ? <div className="muted">Point the camera at a building, solar panel, pole or machine.</div> : (
              <ul className="alert-list">
                {Object.values(seen).sort((a, b) => (a.ts < b.ts ? 1 : -1)).map((a) => (
                  <li key={a.asset_id} className={a.level === 'critical' || a.level === 'priority' ? a.level : ''}>
                    <div className="row between">
                      <strong>{a.asset_type} · {a.firstType === 'new_asset' ? 'NEW (first time seen)' : 'SEEN BEFORE'}</strong>
                      <Link to={`/assets/${a.asset_id}`} className="mono small">{a.asset_id}</Link>
                    </div>
                    {a.history && (
                      <div className="small">
                        Last inspected {when(a.history.last_checked_ts)} · was <strong>{a.history.previous_condition}</strong>
                        {a.history.previous_severity !== null && a.history.previous_severity !== undefined && <> · severity then {fmt(a.history.previous_severity)}</>}
                        {a.history.open_defects?.length ? <> · {a.history.open_defects.length} open defect(s)</> : null}
                      </div>
                    )}
                    <div className="small">
                      Now <strong>{a.label}</strong> ({fmt(a.confidence)}) · severity {fmt(a.severity)} {a.level}
                      {a.change?.since_last_visit
                        ? <> · <strong>{a.change.since_last_visit.significant ? 'CHANGED since last visit' : 'no change since last visit'}</strong></>
                        : a.firstType === 'new_asset' && <> · first inspection</>}
                    </div>
                    {a.context && <div className="small muted">Site context {a.context.level} (K {fmt(a.context.score)}) · criticality ×{fmt(a.context.criticality_multiplier)}</div>}
                  </li>
                ))}
              </ul>
            )}
          </Card>

          <Card title={`Alerts (${alerts.length})`}>
            {alerts.length === 0 ? <div className="muted">No alerts yet.</div> : (
              <ul className="alert-list">
                {alerts.map((a) => (
                  <li key={a.key} className={a.level}>
                    <div className="row between"><strong>{a.label}</strong><span>S {fmt(a.severity)}</span></div>
                    <div className="small"><Link to={`/assets/${a.asset_id}`} className="mono">{a.asset_id}</Link> · {when(a.ts)} · {fmt(a.latency, 0)} ms</div>
                    {a.history && <div className="small muted">was {a.history.previous_condition}{a.history.open_defects?.length ? ` · ${a.history.open_defects.length} open defect(s)` : ''}</div>}
                  </li>
                ))}
              </ul>
            )}
          </Card>
        </div>
      </div>
    </div>
  )
}
