import { useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { errorText, getBriefing, refreshSiteContext, startSession } from '../api'
import { hasRole, useAuth } from '../auth'
import { Card, Empty, ErrorBox, fmt, Loading, Simulated, StatusPill, when } from '../components/ui'
import { useAsync } from '../hooks'

// Device position from the browser (Wi-Fi / GPS); null when denied, unavailable or slow.
export const devicePosition = () => new Promise((resolve) => {
  if (!navigator.geolocation) return resolve(null)
  navigator.geolocation.getCurrentPosition(
    (p) => resolve({ latitude: p.coords.latitude, longitude: p.coords.longitude, accuracy_m: p.coords.accuracy }),
    () => resolve(null),
    { enableHighAccuracy: true, timeout: 8000, maximumAge: 60000 },
  )
})

export default function Briefing() {
  const { siteId } = useParams()
  const brief = useAsync(() => getBriefing(siteId), [siteId])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const [sync, setSync] = useState(null)
  const nav = useNavigate()
  const { user } = useAuth()
  const [ctxBusy, setCtxBusy] = useState(false)

  const refreshContext = async () => {
    setCtxBusy(true)
    setError(null)
    try {
      await refreshSiteContext(siteId)
      brief.reload()
    } catch (e) {
      setError(errorText(e))
    } finally {
      setCtxBusy(false)
    }
  }

  const start = async () => {
    setBusy(true)
    setError(null)
    try {
      const r = await startSession(siteId, await devicePosition())
      setSync(r.sync)
      nav(`/flight/${r.session.session_id}`, { state: { sync: r.sync } })
    } catch (e) {
      setError(errorText(e))
    } finally {
      setBusy(false)
    }
  }

  const b = brief.data
  return (
    <div className="page">
      <header className="page-head">
        <h2>Pre-flight briefing · <span className="mono">{siteId}</span></h2>
        <button className="btn primary" onClick={start} disabled={busy}>{busy ? 'Syncing with cloud…' : 'Start session'}</button>
      </header>
      <p className="muted">Starting a session checks the cloud (FR-1), pulls the latest site memory when online (FR-2) or runs from the local database when offline (FR-3).</p>
      <ErrorBox error={error || brief.error} />
      {sync && <div className="muted">Sync: {sync.mode}</div>}
      {b && (
        <Card
          title={b.context ? `Site context risk · ${b.context.level} (K = ${fmt(b.context.score)})` : 'Site context risk'}
          actions={hasRole(user) && <button className="btn" onClick={refreshContext} disabled={ctxBusy}>{ctxBusy ? 'Fetching live data…' : 'Refresh live data'}</button>}
        >
          {!b.context ? <Empty>No context yet. It is fetched automatically when a session starts: the site is located from this device (allow location in the browser) or, failing that, from the internet connection.</Empty> : (
            <>
              <p className="muted small">
                Criticality ×{fmt(b.context.criticality_multiplier, 3)} · data coverage {Math.round(b.context.coverage * 100)}% · fetched {when(b.context.fetched_ts)} · {b.context.region}
                <br />Location {fmt(b.context.location.latitude, 5)}, {fmt(b.context.location.longitude, 5)} · {{
                  device: `from this device${b.context.location_source?.accuracy_m ? ` (±${Math.round(b.context.location_source.accuracy_m)} m)` : ''}`,
                  ip: `from the internet connection, city level (${b.context.location_source?.place || '?'})`,
                }[b.context.location_source?.source] || 'entered for the site'}
              </p>
              <table className="table">
                <thead><tr><th>Factor</th><th>Score</th><th>Weight</th><th>Evidence</th><th>Source</th></tr></thead>
                <tbody>
                  {Object.entries(b.context.factors).map(([name, f]) => (
                    <tr key={name}>
                      <td>{name.replace(/_/g, ' ')}</td>
                      <td>{f.score === null ? <span className="muted">n/a</span> : fmt(f.score)}</td>
                      <td>{fmt(f.weight)}</td>
                      <td className="small">{f.evidence}</td>
                      <td className="small muted">{f.source}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          )}
        </Card>
      )}
      {brief.loading ? <Loading /> : b && (
        <div className="grid two">
          <Card title={`Assets due for recheck (${b.assets_due.length}), ranked by R`}>
            {b.assets_due.length === 0 ? <Empty>No assets known at this site yet — the first flight registers them.</Empty> : (
              <table className="table">
                <thead><tr><th>#</th><th>Asset</th><th>Type</th><th>Condition</th><th title="R = 0.5·Sr + 0.3·T + 0.2·D + 0.15·K">R</th><th>Sr / T / D</th><th>Last checked</th><th /></tr></thead>
                <tbody>
                  {b.assets_due.map((a, i) => (
                    <tr key={a.asset_id}>
                      <td>{i + 1}</td>
                      <td><Link to={`/assets/${a.asset_id}`} className="mono">{a.asset_id}</Link>{a.marker_id !== null && a.marker_id !== undefined && <span className="muted small"> · ArUco {a.marker_id}</span>}</td>
                      <td>{a.asset_type}</td>
                      <td>{a.current_condition}</td>
                      <td><strong>{fmt(a.revisit_priority)}</strong></td>
                      <td className="small muted">{a.components ? `${fmt(a.components.recent_severity)} / ${fmt(a.components.trend)} / ${fmt(a.components.days_since_inspection)}` : '—'}</td>
                      <td>{when(a.last_checked_ts)}</td>
                      <td><Simulated on={a.simulated} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </Card>
          <Card title={`Unresolved defects (${b.unresolved_defects.length})`}>
            {b.unresolved_defects.length === 0 ? <Empty>None.</Empty> : (
              <table className="table">
                <thead><tr><th>Asset</th><th>Defect</th><th>Severity</th><th>Status</th><th>Last seen</th></tr></thead>
                <tbody>
                  {b.unresolved_defects.map((d) => (
                    <tr key={d.defect_id}>
                      <td className="mono">{d.asset_id}</td><td>{d.defect_type}</td><td>{fmt(d.severity_score)}</td>
                      <td><StatusPill status={d.status} /></td><td>{when(d.last_seen_ts)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </Card>
        </div>
      )}
    </div>
  )
}
