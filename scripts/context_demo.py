"""Context-risk demo on real Chennai locations.

Every number comes from live public data: the place is geocoded with OpenStreetMap
Nominatim, then ml/context_risk.py fetches Open-Meteo weather, the USGS earthquake
catalogue and OpenStreetMap building tags, and applies the BIS IS 1893 seismic zone.

  # context only
  python scripts/context_demo.py --place "Rajiv Gandhi Government General Hospital, Chennai" \
                                 --place "Ambattur Industrial Estate, Chennai"

  # plus: the same asset visited repeatedly at every place (visit 1 registers it, later visits
  # are matched to memory: "last checked …, changed / not changed", severity uses the site context)
  python scripts/context_demo.py --place ... --place ... --image wall_day1.jpg --image wall_day1.jpg --image wall_later.jpg

  # a construction year the public data lacks (place index starts at 1)
  python scripts/context_demo.py --place ... --year 1=1960

Runs against its own database (data/context_demo/) so the app's inspection history is untouched.
"""

from __future__ import annotations

import argparse
import copy
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from ml.config import load_config  # noqa: E402
from ml.context_risk import ContextRiskService, geocode  # noqa: E402


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--place", action="append", required=True, help="real place name in Chennai (repeatable)")
    ap.add_argument("--image", action="append", default=[],
                    help="photo of the asset; repeat for later visits of the SAME asset, in order")
    ap.add_argument("--asset-type", default="building")
    ap.add_argument("--year", action="append", default=[], help="N=YYYY construction year for the N-th place")
    args = ap.parse_args()

    config = copy.deepcopy(load_config())
    demo_dir = ROOT / "data" / "context_demo"
    config["storage"]["db_url"] = f"sqlite:///{(demo_dir / 'demo.db').as_posix()}"
    config["storage"]["images_dir"] = str(demo_dir / "images")
    config["context_risk"]["auto_refresh"] = False
    years = {int(k): int(v) for k, v in (y.split("=") for y in args.year)}

    from memory.store import LocalStore

    store = LocalStore(config)
    context = ContextRiskService(config, store)
    sites = []
    for i, place in enumerate(args.place, start=1):
        g = geocode(place, config)
        time.sleep(1.1)   # Nominatim usage policy: at most one request per second
        site_id = f"chennai_{slug(place)}"
        store.ensure_site(site_id, place, g["latitude"], g["longitude"], enqueue=False)
        if i in years:
            context.set_attributes(site_id, {"construction_year": years[i]})
        print(f"\n[{i}] {place}\n    {g['display_name']}\n    lat {g['latitude']:.5f}, lon {g['longitude']:.5f}")
        r = context.refresh(site_id, force=True)
        print(f"    context K = {r['score']:.3f} ({r['level']}), criticality x{r['criticality_multiplier']:.3f}, "
              f"data coverage {r['coverage']:.0%}")
        for name, f in r["factors"].items():
            val = "  n/a" if f["score"] is None else f"{f['score']:.3f}"
            print(f"      {name:16} {val}  w={f['weight']:.2f}  {f['evidence']}  [{f['source']}]")
        sites.append((site_id, place))

    if args.image:
        from api.services import build_services

        services = build_services(config, store=store, use_cloud=False)
        services.engine.context = context
        for site_id, place in sites:
            print(f"\n=== {place} ===")
            for visit, path in enumerate(args.image, start=1):
                sid = store.create_session(site_id, "demo", "offline", {})["session_id"]
                ev = services.engine.observe(Image.open(path).convert("RGB"), args.asset_type, sid)
                store.update_session(sid, status="ended")
                if ev.type == "error":
                    print(f"  visit {visit}: error: {ev.error}")
                    continue
                crit = store.observations(ev.asset_id)[-1]["metadata"]["criticality"]
                print(f"  visit {visit}: {Path(path).name}")
                if ev.type == "new_asset":
                    print(f"    new asset {ev.asset_id} registered")
                else:
                    h = ev.history
                    print(f"    recognised {ev.asset_id} ({ev.identity['stage']} match); last checked {h['last_checked_ts']}, "
                          f"was '{h['previous_condition']}'")
                verdict = "significant change" if ev.change["significant"] else "no significant change"
                print(f"    now '{ev.condition['label']}' ({ev.condition['confidence']:.2f}); {verdict}: "
                      f"M={ev.change['change_magnitude']:.3f}  C={crit:.3f}  S={ev.severity_score:.3f} ({ev.severity_level})  "
                      f"revisit={ev.revisit['revisit_priority']:.3f}")
        print("\nM = change since last visit, C = asset criticality after site context, S = (w1*M + w2*C)*Cf, "
              "S = 0 when M is below the change threshold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
