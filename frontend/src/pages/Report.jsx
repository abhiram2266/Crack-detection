import { useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { errorText, getReport, getReportFile, imageUrl } from '../api'
import { Card, Empty, ErrorBox, fmt, Loading, Severity, Simulated, Stat, StatusPill, when } from '../components/ui'
import { useAsync } from '../hooks'

function Rows({ items, flagged }) {
  if (!items.length) return <Empty>None.</Empty>
  return (
    <table className="table">
      <thead>
        <tr><th /><th>Asset</th><th>Type</th><th>Condition</th>{flagged && <th>Severity</th>}<th>Change M</th><th>Cf</th><th>Match</th><th>R</th><th>Time</th></tr>
      </thead>
      <tbody>
        {items.map((e) => (
          <tr key={`${e.asset_id}-${e.observation_ts}`}>
            <td>{e.image_reference && <img src={imageUrl(e.image_reference)} alt="" className="thumb" />}</td>
            <td><Link to={`/assets/${e.asset_id}`} className="mono">{e.asset_id}</Link></td>
            <td>{e.asset_type}</td>
            <td>{e.condition}{e.defect_type ? ` (${e.defect_type})` : ''}</td>
            {flagged && <td><Severity score={e.severity_score} level={e.severity_level} /></td>}
            <td>{fmt(e.change_magnitude)}</td><td>{fmt(e.observation_confidence)}</td><td>{e.match_stage}</td>
            <td>{fmt(e.revisit_priority)}</td><td>{when(e.observation_ts)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

export default function Report() {
  const { sessionId } = useParams()
  const rep = useAsync(() => getReport(sessionId), [sessionId])
  const [error, setError] = useState(null)

  const download = async (format) => {
    try {
      const blob = await getReportFile(sessionId, format)
      const url = URL.createObjectURL(blob)
      if (format === 'html') {
        window.open(url, '_blank', 'noopener')
      } else {
        const a = document.createElement('a')
        a.href = url
        a.download = `audi_report_${sessionId}.pdf`
        a.click()
      }
      setTimeout(() => URL.revokeObjectURL(url), 60000)
    } catch (e) { setError(errorText(e)) }
  }

  if (rep.loading) return <Loading />
  const r = rep.data
  return (
    <div className="page">
      <header className="page-head">
        <h2>Post-flight report <span className="mono small">{sessionId}</span></h2>
        <div className="row gap">
          <button className="btn" onClick={() => download('pdf')}>Download PDF</button>
          <button className="btn ghost" onClick={() => download('html')}>Open HTML</button>
        </div>
      </header>
      <ErrorBox error={error || rep.error} />
      {r && (
        <>
          <p className="muted">
            Site <span className="mono">{r.session.site_id}</span> · operator {r.session.operator_id} · {r.session.mission_date} · <StatusPill status={r.session.mode} /> <Simulated on={r.session.simulated} />
          </p>
          <div className="stats">
            <Stat label="Assets processed" value={r.counts.assets_processed} />
            <Stat label="Flagged" value={r.counts.flagged} tone={r.counts.flagged ? 'crit' : ''} />
            <Stat label="Confirmed" value={r.counts.confirmed} />
            <Stat label="New assets" value={r.counts.new_assets} />
            <Stat label="Seen before" value={r.counts.revisited ?? 0} />
            <Stat label="Open defects" value={r.counts.open_defects} />
            <Stat label="Escalations" value={r.counts.escalations} tone={r.counts.escalations ? 'warn' : ''} />
          </div>
          {r.context && (
            <Card title={`Site context risk · ${r.context.level} (K = ${fmt(r.context.score)})`}>
              <p className="small muted">Criticality ×{fmt(r.context.criticality_multiplier, 3)} · data coverage {Math.round(r.context.coverage * 100)}%</p>
              <ul className="small">{r.context.reasons.map((x) => <li key={x}>{x}</li>)}</ul>
            </Card>
          )}
          <Card title="Re-inspected assets: this flight vs their previous inspection">
            {!(r.revisited_assets || []).length ? <Empty>No asset from an earlier flight was seen.</Empty> : (
              <table className="table">
                <thead><tr><th /><th>Asset</th><th>Type</th><th>Previous inspection</th><th>Then</th><th>Now</th><th>Severity then → now</th><th>Inspections</th></tr></thead>
                <tbody>
                  {r.revisited_assets.map((a) => (
                    <tr key={a.asset_id}>
                      <td>{a.image_reference && <img src={imageUrl(a.image_reference)} alt="" className="thumb" />}</td>
                      <td><Link to={`/assets/${a.asset_id}`} className="mono">{a.asset_id}</Link></td>
                      <td>{a.asset_type}</td>
                      <td>{when(a.previous_inspection_ts)}</td>
                      <td>{a.previous_condition}</td>
                      <td><strong>{a.condition}</strong>{a.condition_changed && <span className="pill warn"> changed</span>}</td>
                      <td>{fmt(a.previous_severity)} → <Severity score={a.severity_score} level={a.severity_level} /></td>
                      <td>{a.inspections_total}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </Card>
          <Card title="Flagged defects (by severity)"><Rows items={r.flagged_defects} flagged /></Card>
          <Card title="Newly registered assets"><Rows items={r.new_assets} /></Card>
          <Card title="Confirmed assets (no significant change)"><Rows items={r.confirmed_assets} /></Card>
          <div className="grid two">
            <Card title="Open defects">
              {r.open_defects.length === 0 ? <Empty>None.</Empty> : (
                <table className="table">
                  <thead><tr><th>Asset</th><th>Defect</th><th>Severity</th><th>Status</th></tr></thead>
                  <tbody>{r.open_defects.map((d) => (
                    <tr key={d.defect_id}><td className="mono">{d.asset_id}</td><td><Link to={`/defects/${d.defect_id}`}>{d.defect_type}</Link></td><td>{fmt(d.severity_score)}</td><td><StatusPill status={d.status} /></td></tr>
                  ))}</tbody>
                </table>
              )}
            </Card>
            <Card title="Escalations & sync">
              {(r.escalations || []).length === 0 ? <div className="muted">No escalations.</div> : (
                <ul className="plain">{r.escalations.map((e) => <li key={e.notification_id}>{e.username}: {e.message}</li>)}</ul>
              )}
              <pre className="pre small">{JSON.stringify(r.sync, null, 2)}</pre>
            </Card>
          </div>
        </>
      )}
    </div>
  )
}
