import { useState } from 'react'
import { Link } from 'react-router-dom'
import { analyzeImage, errorText, getSessions, getSites, imageUrl, searchMemory } from '../api'
import { Card, ErrorBox, Field, fmt, Severity, when } from '../components/ui'
import { useAsync } from '../hooks'

const ASSET_TYPES = ['solar_panel', 'building', 'utility_pole', 'machinery']

export default function SemanticMemory() {
  const sites = useAsync(getSites, [])
  const sessions = useAsync(() => getSessions(), [])
  const [file, setFile] = useState(null)
  const [assetType, setAssetType] = useState('solar_panel')
  const [siteId, setSiteId] = useState('')
  const [sessionId, setSessionId] = useState('')
  const [addToMemory, setAddToMemory] = useState(false)
  const [result, setResult] = useState(null)
  const [matches, setMatches] = useState(null)
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)
  const active = (sessions.data || []).filter((s) => s.status === 'active')

  const run = async (fn) => {
    setBusy(true)
    setError(null)
    try { await fn() } catch (e) { setError(errorText(e)) } finally { setBusy(false) }
  }
  const analyse = () => run(async () => setResult(await analyzeImage(file, assetType, sessionId || null, addToMemory)))
  const search = () => run(async () => setMatches(await searchMemory(file, assetType, siteId || null, 8)))

  return (
    <div className="page">
      <header className="page-head"><h2>Analyse an image &amp; search memory</h2></header>
      <Card>
        <div className="form-grid">
          <Field label="Close-up image of one asset"><input type="file" accept="image/*" onChange={(e) => setFile(e.target.files[0] || null)} /></Field>
          <Field label="Asset type">
            <select value={assetType} onChange={(e) => setAssetType(e.target.value)}>{ASSET_TYPES.map((t) => <option key={t}>{t}</option>)}</select>
          </Field>
          <Field label="Site (search)">
            <select value={siteId} onChange={(e) => setSiteId(e.target.value)}>
              <option value="">Default site</option>
              {(sites.data || []).map((s) => <option key={s.site_id} value={s.site_id}>{s.name}</option>)}
            </select>
          </Field>
          <Field label="Session (to store the result)" hint="Only an active session can store observations (D7).">
            <select value={sessionId} onChange={(e) => { setSessionId(e.target.value); if (!e.target.value) setAddToMemory(false) }}>
              <option value="">None — analyse only</option>
              {active.map((s) => <option key={s.session_id} value={s.session_id}>{s.site_id} · {s.session_id}</option>)}
            </select>
          </Field>
          <label className="row gap"><input type="checkbox" checked={addToMemory} disabled={!sessionId} onChange={(e) => setAddToMemory(e.target.checked)} /> add to memory</label>
        </div>
        <div className="row gap">
          <button className="btn primary" disabled={!file || busy} onClick={analyse}>Analyse</button>
          <button className="btn" disabled={!file || busy} onClick={search}>Find similar assets</button>
        </div>
        <ErrorBox error={error} />
      </Card>

      {result && (
        <Card title={`Result: ${result.type}${result.persisted ? ' (stored)' : ' (not stored)'}`}>
          <dl className="kv">
            <dt>Condition</dt><dd>{result.condition.label} ({fmt(result.condition.confidence)})</dd>
            <dt>Severity</dt><dd><Severity score={result.severity_score} level={result.severity_level} /></dd>
            <dt>Identity</dt><dd>{result.asset_id ? <Link to={`/assets/${result.asset_id}`} className="mono">{result.asset_id}</Link> : 'new'} · {result.identity?.stage}</dd>
            {result.history && <><dt>Memory</dt><dd>seen before · last checked {when(result.history.last_checked_ts)} · was <strong>{result.history.previous_condition}</strong></dd></>}
            {result.context && <><dt>Site context</dt><dd>{result.context.level} (K = {fmt(result.context.score)}) · criticality ×{fmt(result.context.criticality_multiplier, 3)}<div className="small muted">{result.context.reasons.join(' · ')}</div></dd></>}
            {result.change && <><dt>Change M</dt><dd>{fmt(result.change.change_magnitude)} (condition {fmt(result.change.condition_delta)}, visual {fmt(result.change.visual_delta)})</dd></>}
            <dt>Class scores</dt>
            <dd className="small">{Object.entries(result.condition.scores || {}).sort((a, b) => b[1] - a[1]).map(([k, v]) => `${k} ${fmt(v)}`).join(' · ')}</dd>
          </dl>
        </Card>
      )}

      {matches && (
        <Card title={`Most similar stored assets (${matches.length}) — global embedding only`}>
          <div className="match-grid">
            {matches.map((m) => (
              <Link key={m.asset.asset_id} to={`/assets/${m.asset.asset_id}`} className="match">
                <span className="mono small">{m.asset.asset_id}</span>
                <span>{m.asset.current_condition}</span>
                <span className="muted small">cosine {fmt(m.similarity, 3)}</span>
              </Link>
            ))}
          </div>
        </Card>
      )}
      {result?.image_reference && <img src={imageUrl(result.image_reference)} alt="Stored crop" className="thumb-lg" />}
    </div>
  )
}
