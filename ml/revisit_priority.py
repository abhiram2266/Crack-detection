"""Revisit priority, SRS §8.2 with the amendments in docs/SRS_AMENDMENTS.md.

R = w3·Sr + w4·T + w5·D + w6·K
  Sr latest severity of the asset.
  T  severity trend over the last N missions (revisit_priority.trend_window):
     least-squares slope of per-mission severity (severity per mission index)
     multiplied by (n − 1), i.e. the fitted rise across the window, clipped to
     [0, 1]. Falling or flat trends give 0 (D15).
  D  min(1, days since last inspection / max_days_since_inspection), evaluated
     at query time ("now"), so it keeps growing between missions (D9).
  K  context score of the asset's site (ml/context_risk.py), 0 when unknown.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional, Sequence

from ml.config import cfg


def parse_ts(ts: str | datetime) -> datetime:
    dt = ts if isinstance(ts, datetime) else datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def calculate_trend(mission_severities: Sequence[float], window: int) -> float:
    scores = [float(s) for s in mission_severities][-window:]
    n = len(scores)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(scores) / n
    num = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(scores))
    den = sum((i - x_mean) ** 2 for i in range(n))
    slope = num / den
    return max(0.0, min(1.0, slope * (n - 1)))


def time_decay(last_inspection: Optional[str | datetime], now: Optional[datetime], max_days: float) -> float:
    if last_inspection is None:
        return 1.0
    now = now or datetime.now(timezone.utc)
    days = max(0.0, (now - parse_ts(last_inspection)).total_seconds() / 86400.0)
    return min(1.0, days / max_days)


def compute_revisit(
    mission_severities: Sequence[float],
    last_inspection: Optional[str | datetime],
    config: Dict,
    now: Optional[datetime] = None,
    context_score: float = 0.0,
) -> Dict[str, float]:
    rp = cfg(config, "revisit_priority")
    sr = max(0.0, min(1.0, float(mission_severities[-1]) if mission_severities else 0.0))
    t = calculate_trend(mission_severities, int(rp["trend_window"]))
    d = time_decay(last_inspection, now, float(rp["max_days_since_inspection"]))
    k = max(0.0, min(1.0, float(context_score)))
    r = float(rp["w3"]) * sr + float(rp["w4"]) * t + float(rp["w5"]) * d + float(rp.get("w6", 0.0)) * k
    return {
        "recent_severity": sr,
        "trend": t,
        "days_since_inspection": d,
        "context": k,
        "revisit_priority": max(0.0, min(1.0, r)),
    }
