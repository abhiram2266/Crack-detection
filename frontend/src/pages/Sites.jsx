import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { endSession, errorText, getInterrupted, getSessions, getSites, quickStart, resumeSession } from '../api'
import { devicePosition } from './Briefing'
import { Card, ErrorBox, StatusPill, when } from '../components/ui'
import { useAsync } from '../hooks'

// Only sessions from this point on are listed; older test/demo sessions stay in the database but are hidden.
const SHOW_FROM = '2026-10-15T16:00:00'

export default function Sites() {
  const interrupted = useAsync(getInterrupted, [])
  const sites = useAsync(getSites, [])
  const sessions = useAsync(() => getSessions(), [])
  const [error, setError] = useState(null)
  const nav = useNavigate()
  const [starting, setStarting] = useState(null)

  const quick = async () => {
    setError(null)
    try {
      setStarting('Finding your location…')
      const pos = await devicePosition()
      setStarting(pos ? 'Starting session…' : 'No device location; using the internet connection…')
      const r = await quickStart(pos)
      nav(`/flight/${r.session.session_id}`, { state: { sync: r.sync, quickStart: r.quick_start } })
    } catch (e) {
      setError(errorText(e))
    } finally {
      setStarting(null)
    }
  }

  const resume = async (id) => {
    try {
      await resumeSession(id)
      nav(`/flight/${id}`)
    } catch (e) { setError(errorText(e)) }
  }
  const close = async (id) => {
    try {
      await endSession(id)
      nav(`/report/${id}`)
    } catch (e) { setError(errorText(e)) }
  }

  const unfinished = (interrupted.data || []).filter((x) => x.started_ts >= SHOW_FROM)
  const names = Object.fromEntries((sites.data || []).map((x) => [x.site_id, x.name]))
  // Locations you have finished an inspection at (newest first).
  const inspected = []
  for (const x of sessions.data || []) {
    if (x.status !== 'ended' || x.started_ts < SHOW_FROM || inspected.some((i) => i.site_id === x.site_id)) continue
    inspected.push(x)
  }

  return (
    <div className="page">
      <header className="page-head"><h2>Start an inspection</h2></header>
      <Card title="Quick start">
        <div className="row between wrap gap">
          <p className="muted" style={{ margin: 0 }}>
            Uses this device's location: continues the site you inspected here before (so assets are recognised),
            or creates a new one named after the place. Rainfall, seismic and building data are fetched automatically.
          </p>
          <button className="btn primary" onClick={quick} disabled={!!starting}>{starting || 'Start new session here'}</button>
        </div>
      </Card>
      <ErrorBox error={error || interrupted.error} />
      {unfinished.length > 0 && (
        <Card title="Unfinished sessions (crash recovery)" className="warn-card">
          <table className="table">
            <thead><tr><th>Session</th><th>Site</th><th>Started</th><th>Status</th><th>Assets</th><th /></tr></thead>
            <tbody>
              {unfinished.map((s) => (
                <tr key={s.session_id}>
                  <td className="mono">{s.session_id}</td><td>{s.site_id}</td><td>{when(s.started_ts)}</td>
                  <td><StatusPill status={s.status} /></td><td>{s.assets_processed}</td>
                  <td className="row gap">
                    <button className="btn small" onClick={() => resume(s.session_id)}>Resume</button>
                    <button className="btn small ghost" onClick={() => close(s.session_id)}>Close &amp; report</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}
      {inspected.length > 0 && (
        <Card title="Your inspected locations">
          <div className="site-grid">
            {inspected.map((x) => (
              <Link key={x.site_id} to={`/report/${x.session_id}`} className="site-tile">
                <strong>{names[x.site_id] || x.site_id}</strong>
                <span className="muted small">Last inspected {when(x.ended_ts || x.started_ts)}</span>
              </Link>
            ))}
          </div>
        </Card>
      )}
    </div>
  )
}
