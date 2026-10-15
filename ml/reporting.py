"""Post-flight report (FR-18): JSON, HTML and PDF renderings."""

from __future__ import annotations

import html
import io
from typing import Any, Dict, List

from memory.store import LocalStore


def build_report(store: LocalStore, session_id: str, sync_result: Dict[str, Any], escalations: List[Dict]) -> Dict[str, Any]:
    session = store.get_session(session_id)
    if session is None:
        raise KeyError(session_id)
    observations = store.session_observations(session_id)

    latest: Dict[str, Dict[str, Any]] = {}
    for o in observations:
        latest[o["asset_id"]] = o
    worst: Dict[str, Dict[str, Any]] = {}
    for o in observations:
        if o["asset_id"] not in worst or o["severity_score"] > worst[o["asset_id"]]["severity_score"]:
            worst[o["asset_id"]] = o

    assets = {a: store.get_asset(a) for a in latest}

    # What memory held before this flight: each asset's last observation from an earlier session.
    previous: Dict[str, Dict[str, Any]] = {}
    for aid in latest:
        earlier = [o for o in store.observations(aid) if o["session_id"] != session_id and o["observation_ts"] < session["started_ts"]]
        if earlier:
            previous[aid] = max(earlier, key=lambda o: o["observation_ts"])

    def entry(o: Dict[str, Any]) -> Dict[str, Any]:
        a = assets[o["asset_id"]] or {}
        return {
            "asset_id": o["asset_id"],
            "asset_type": a.get("asset_type"),
            "condition": o["condition_description"],
            "defect_type": o["defect_type"],
            "severity_score": round(o["severity_score"], 4),
            "severity_level": o["severity_level"],
            "change_magnitude": round(o["change_magnitude"], 4),
            "observation_confidence": round(o["observation_confidence"], 4),
            "match_stage": o["match_stage"],
            "observation_ts": o["observation_ts"],
            "image_reference": o["image_reference"],
            "heatmap_reference": o["heatmap_reference"],
            "revisit_priority": round(a.get("revisit_priority_score", 0.0), 4),
        }

    flagged = sorted(
        (entry(o) for o in worst.values() if o["status"] == "changed" or (o["status"] == "new" and o["severity_score"] > 0)),
        key=lambda e: e["severity_score"],
        reverse=True,
    )
    confirmed = [entry(o) for o in latest.values() if o["status"] == "confirmed" and o["asset_id"] not in {f["asset_id"] for f in flagged}]
    new_assets = [entry(o) for o in observations if o["status"] == "new"]

    revisited = []
    for aid, o in latest.items():
        p = previous.get(aid)
        if p is None:
            continue
        revisited.append({
            **entry(o),
            "previous_inspection_ts": p["observation_ts"],
            "previous_condition": p["condition_description"],
            "previous_severity": round(p["severity_score"], 4),
            "inspections_total": len(store.observations(aid)),
            "condition_changed": p["condition_description"] != o["condition_description"],
            "severity_delta": round(o["severity_score"] - p["severity_score"], 4),
        })
    revisited.sort(key=lambda e: e["severity_delta"], reverse=True)

    defects = []
    for aid in latest:
        for d in store.list_defects(asset_id=aid):
            if d["status"] != "Resolved":
                defects.append(
                    {k: d[k] for k in ("defect_id", "asset_id", "defect_type", "severity_score", "max_severity", "status", "first_detected_ts", "resolution_suggested")}
                )

    # Site context in force during the flight (snapshot stored with each observation).
    context = next((o["metadata"].get("context") for o in reversed(observations) if o["metadata"].get("context")), None)

    return {
        "session": {k: session[k] for k in ("session_id", "site_id", "operator_id", "mission_date", "started_ts", "ended_ts", "mode", "assets_processed", "simulated")},
        "counts": {
            "observations": len(observations),
            "assets_processed": len(latest),
            "flagged": len(flagged),
            "confirmed": len(confirmed),
            "new_assets": len(new_assets),
            "revisited": len(revisited),
            "open_defects": len(defects),
            "escalations": len(escalations),
        },
        "flagged_defects": flagged,
        "severity_ranking": [{"asset_id": f["asset_id"], "severity_score": f["severity_score"], "severity_level": f["severity_level"]} for f in flagged],
        "confirmed_assets": confirmed,
        "new_assets": new_assets,
        "revisited_assets": revisited,
        "open_defects": defects,
        "escalations": escalations,
        "context": context,
        "sync": sync_result,
    }


def _rows(items: List[Dict[str, Any]], cols: List[str]) -> str:
    if not items:
        return "<p><em>None</em></p>"
    head = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(i.get(c, '')))}</td>" for c in cols) + "</tr>" for i in items)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


FLAG_COLS = ["asset_id", "asset_type", "condition", "severity_score", "severity_level", "change_magnitude", "match_stage"]
ASSET_COLS = ["asset_id", "asset_type", "condition", "observation_ts", "revisit_priority"]
REVISIT_COLS = ["asset_id", "asset_type", "previous_inspection_ts", "previous_condition", "condition", "previous_severity", "severity_score", "severity_delta", "inspections_total"]
DEFECT_COLS = ["defect_id", "asset_id", "defect_type", "severity_score", "status", "resolution_suggested"]


def render_html(report: Dict[str, Any]) -> str:
    s = report["session"]
    c = report["counts"]
    ctx = report.get("context")
    ctx_html = (
        f"<h2>Site context risk</h2><p>{html.escape(ctx['level'])} · K = {ctx['score']:.2f} · criticality ×{ctx['criticality_multiplier']:.2f}"
        f" · data coverage {ctx['coverage']:.0%}</p><ul>" + "".join(f"<li>{html.escape(r)}</li>" for r in ctx["reasons"]) + "</ul>"
        if ctx
        else ""
    )
    sim = "<p><strong>Simulated session</strong> — data generated for demonstration.</p>" if s.get("simulated") else ""
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>AUDI report {html.escape(s['session_id'])}</title>
<style>body{{font-family:system-ui,sans-serif;margin:2rem;color:#1d2433}}table{{border-collapse:collapse;width:100%;margin-bottom:1.5rem}}
th,td{{border:1px solid #d5dae3;padding:4px 8px;font-size:13px;text-align:left}}th{{background:#f2f4f8}}</style></head><body>
<h1>Post-flight report</h1>{sim}
<p>Session <code>{html.escape(s['session_id'])}</code> · site <code>{html.escape(s['site_id'])}</code> · operator {html.escape(str(s['operator_id']))} ·
{html.escape(s['mission_date'])} · mode {html.escape(s['mode'])}</p>
<p>Assets processed {c['assets_processed']} · flagged {c['flagged']} · confirmed {c['confirmed']} · new {c['new_assets']} · open defects {c['open_defects']} · escalations {c['escalations']}</p>
{ctx_html}
<h2>Flagged defects (by severity)</h2>{_rows(report['flagged_defects'], FLAG_COLS)}
<h2>Re-inspected assets (compared with their previous inspection)</h2>{_rows(report.get('revisited_assets', []), REVISIT_COLS)}
<h2>Newly registered assets</h2>{_rows(report['new_assets'], ASSET_COLS)}
<h2>Confirmed assets</h2>{_rows(report['confirmed_assets'], ASSET_COLS)}
<h2>Open defects</h2>{_rows(report['open_defects'], DEFECT_COLS)}
<h2>Sync</h2><pre>{html.escape(str(report['sync']))}</pre></body></html>"""


def render_pdf(report: Dict[str, Any]) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4), title=f"AUDI report {report['session']['session_id']}")
    styles = getSampleStyleSheet()
    s, c = report["session"], report["counts"]
    story = [Paragraph("AUDI post-flight report", styles["Title"])]
    if s.get("simulated"):
        story.append(Paragraph("<b>Simulated session</b> — data generated for demonstration.", styles["Normal"]))
    story.append(
        Paragraph(
            f"Session {s['session_id']} · site {s['site_id']} · operator {s['operator_id']} · {s['mission_date']} · mode {s['mode']}",
            styles["Normal"],
        )
    )
    story.append(
        Paragraph(
            f"Assets processed {c['assets_processed']} · flagged {c['flagged']} · confirmed {c['confirmed']} · "
            f"new {c['new_assets']} · open defects {c['open_defects']} · escalations {c['escalations']}",
            styles["Normal"],
        )
    )

    def table(title: str, items: List[Dict[str, Any]], cols: List[str]):
        story.append(Spacer(1, 10))
        story.append(Paragraph(title, styles["Heading2"]))
        if not items:
            story.append(Paragraph("None", styles["Italic"]))
            return
        data = [cols] + [[str(i.get(col, "")) for col in cols] for i in items]
        t = Table(data, repeatRows=1)
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f2f4f8")),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d5dae3")),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                ]
            )
        )
        story.append(t)

    table("Flagged defects (by severity)", report["flagged_defects"], FLAG_COLS)
    table("Re-inspected assets (vs previous inspection)", report.get("revisited_assets", []), ["asset_id", "previous_inspection_ts", "previous_condition", "condition", "previous_severity", "severity_score"])
    table("Newly registered assets", report["new_assets"], ASSET_COLS)
    table("Confirmed assets", report["confirmed_assets"], ASSET_COLS)
    table("Open defects", report["open_defects"], DEFECT_COLS)
    doc.build(story)
    return buf.getvalue()
