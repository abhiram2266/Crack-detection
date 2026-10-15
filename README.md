# AUDI — persistent semantic memory for drone inspection

AUDI remembers every physical asset it has inspected (solar panels, utility poles and
insulators, building/concrete surfaces, machinery). On each flight it re-identifies the
assets it sees, compares their condition with the last inspection, scores the change, alerts
the operator live and escalates critical defects to maintenance. It also ranks which assets
to revisit next.

The requirements are the AUDI SRS (FR-1 – FR-20, NFR-1 – NFR-3).
`docs/SRS_TRACEABILITY.md` maps every requirement to code and tests, and
`docs/SRS_AMENDMENTS.md` lists the changes the SRS document itself needs. `PROGRESS.md`
is the verified work log.

## Architecture

```
 camera / RTSP / video file ──┐            browser camera (getUserMedia) ──┐
                              ▼                                            ▼
                    ml/live/camera.py  CameraReader (reconnect, latest-frame-only)
                              │
                    ml/live/pipeline.py  frame skip → duplicate-frame skip (D8)
                              │
          ┌───────────────────┴──────────── SemanticMemoryEngine.process_frame (ml/semantic_memory.py)
          │  detect        ml/detector.py         YOLO-World (open vocabulary) | full_frame
          │  track         ml/tracker.py          IoU + embedding + thumbnail NCC, one identity run per track
          │  embed         ml/embedding_model.py  DINOv2 ViT-S/14 (CLS + patch tokens)
          │  identity      ml/asset_identity.py   ArUco → location gate → top-k (memory/vector_memory.py)
          │                                       → SIFT + RANSAC verification → visual-only → new asset
          │  condition     ml/condition_classifier.py  per-asset-type heads on frozen DINOv2
          │  change        ml/change_detector.py  M = α·condition change + (1−α)·patch-token change
          │  severity      ml/severity.py         S = (w1·M + w2·C)·Cf, S = 0 if M below threshold
          │  store         memory/store.py        SQLite (WAL) + outbox, one transaction per observation
          └──────────────────────────────────────────────┬──────────────────────────────
                                                         │
   api/ (FastAPI, JWT + roles)  ── REST, /ws/live, /ws/notifications, MJPEG /api/live/stream
        │   ml/mission.py      session start/end, briefing (FR-4), revisit priority (ml/revisit_priority.py)
        │   ml/reporting.py    post-flight report JSON/HTML/PDF (FR-18)
        │   ml/escalation.py   critical → subscribed maintenance users only (FR-19)
        │   memory/sync.py     background outbox flush, 10 s bound on start/end sync (NFR-3)
        ▼
   frontend/ (React + Vite)                     cloud/ (separate FastAPI service, own DB, JWT)
   operator · maintenance · admin views         pull_site (FR-2), push_outbox (FR-6/7)
```

Module boundaries are interfaces, so implementations can be swapped. For example, the embedder
can be DINOv2, ResNet or a projection embedder; the detector can be YOLO-World or full-frame;
the vector backend can be NumPy or FAISS (opt-in); and the cloud store is an HTTP client.
Every tunable number is in `config.yaml`.

## Quick start: test cracks (no dataset or training needed)

The trained classifiers (`models/condition_classifier.pt`, `models/asset_type_classifier.npz`)
and the YOLO-World weights are committed, so you can skip `make data` and `make train`.
DINOv2 and CLIP download automatically on first start (~450 MB, needs internet once).

```bash
# Windows (PowerShell): python -m venv .venv; .venv\Scripts\pip install -r requirements.txt
make setup                              # .venv + Python deps + frontend deps
.venv/bin/python scripts/seed_users.py  # creates admin / operator / maintenance users
# passwords are in data/credentials.env (or set AUDI_ADMIN_PASSWORD etc. before seeding)
.venv/bin/python run_api.py             # terminal 1 → API on http://127.0.0.1:8000
cd frontend && npm run dev              # terminal 2 → UI on http://localhost:5173
```

Sign in, open **Preflight → Start new session here** (allow location access), point the
camera at a crack, then end the session to get the report. Severity scales with the crack's
size (`severity.extent` in `config.yaml`); re-inspecting the same crack in a later session
shows it under "Re-inspected assets" with the previous severity.

## Setup

Requires Python 3.11 or 3.12, Node 18+, and macOS, Linux or Windows. `auto` device selection
uses CUDA, then Apple MPS, then CPU.

```bash
make setup          # .venv + pip install -r requirements.txt + npm ci in frontend/
```

### Kaggle

Put your `kaggle.json` (from kaggle.com → Settings → API → Create New Token) in the repo
root. It is git-ignored. The download script sets `KAGGLE_CONFIG_DIR` to the repo root at
runtime and `chmod 600`s the file. It never copies or prints the file.

```bash
make data           # scripts/download_datasets.py + scripts/prepare_unified_dataset.py
python scripts/download_datasets.py --list               # sources and enabled/optional state
python scripts/download_datasets.py --include-optional   # also the large optional sets
```

`dataset/README.md` (generated) lists every source, its licence as stated on Kaggle, the
label mapping and limitations. `dataset/processed/STATS.md` has the per-class counts and the
de-duplication report.

### Camera (Philips or any UVC webcam)

```bash
make camera-probe   # lists cameras and writes reports/camera_probe/cam_<i>.jpg per index
```

Look at the snapshots to find the external camera. Then set `camera.source` in `config.yaml`
to its index, a unique part of its device name, an `rtsp://` / `http://` URL, or a video file
path. On macOS, the terminal running the API needs camera permission (System Settings →
Privacy & Security → Camera). The camera can be unplugged and plugged back in while
running; the reader reconnects every `camera.reconnect_seconds`.

## Run order

```bash
make data           # datasets
make train          # training/train_condition_classifier.py → models/condition_classifier.pt
make eval           # training/evaluate.py → reports/eval/summary.md
make memory         # migrate legacy JSON store, seed users, build simulated history site
make cloud          # cloud service on 127.0.0.1:8100   (terminal 1)
make api            # API on 127.0.0.1:8000             (terminal 2)
make web            # UI on http://localhost:5173       (terminal 3)
```

`make memory` runs `scripts/seed_users.py`, which creates `admin`, `operator`,
`maintenance` and the cloud `sync` user. Passwords come from `AUDI_<ROLE>_PASSWORD` if set.
Otherwise they are generated into `data/credentials.env` (mode 600, never printed), so read
them from that file. The JWT secret comes from `AUDI_JWT_SECRET` or is generated into
`data/.jwt_secret`. CORS allows only `api.cors_origins`.

Checks: `make test` (pytest), `make bench` (NFR benchmarks → `reports/bench.md`) and
`make smoke`. The smoke test starts cloud and API on free ports with throw-away databases,
flies two missions from video files through the live pipeline, and checks briefing,
alerts, report, escalation and sync.

## Live demo script

1. **Prepare.** Print an ArUco marker from `DICT_4X4_50` (any id) and tape it to a desk
   object, for example a box. Set `camera.source` (see Camera). For a single object filling
   the view, set `detection.mode: full_frame` and `detection.full_frame_asset_type` to the
   type to score it as (`machinery` suits most desk objects). With YOLO-World, use the
   configured prompts.
2. **Start services.** `make cloud`, `make api`, `make web`. Sign in as **admin**, then in
   *Sites* create a site and subscribe the maintenance user. In *Config*, show that
   criticality and weights are editable and versioned.
3. **Mission 1.** Sign in as **operator** → *Sites* → pick the site → *Briefing* (empty on the
   first flight) → *Start flight*. Show the object to the camera. The edge-bar overlay shows
   the box, a new asset id and its condition, and the status bar shows fps and latency. Press
   *End session*; the report lists the new assets.
4. **Mission 2.** Draw a "crack" on the object with a marker, or add tape or dust. Open
   *Briefing*: the asset is listed with its revisit priority. Start the flight and show the
   object. The marker re-identifies it, a change alert appears at the screen edge with a
   beep, and a critical alert also shows a red border. End the session and download the PDF
   report.
5. **Escalation.** Sign in as **maintenance**. The inbox shows the escalation (pushed live).
   Open the defect, move it to *In Progress*, and show the asset timeline, severity trend,
   image history and change heatmap.
6. **Offline.** Stop `make cloud`, then start a session: it starts offline within the
   10 s bound. Restart the cloud; the outbox flushes (admin → *Metrics* / sync status).

## Accuracy (from `reports/eval/summary.md`)

Test split, n = 1731, backbone DINOv2 ViT-S/14. *Baseline* is the previous approach,
nearest-neighbour label copy from training embeddings. *Classifier* is the trained
per-asset-type heads. *Webcam* means each test image was degraded once (blur, JPEG,
colour cast, perspective, crop).

| model | accuracy (clean) | macro-F1 (clean) | accuracy (webcam) | macro-F1 (webcam) |
|---|---|---|---|---|
| baseline | 0.929 | 0.873 | 0.908 | 0.857 |
| classifier | **0.947** | **0.921** | **0.945** | **0.933** |

| asset type | baseline macro-F1 | classifier macro-F1 | classifier defect recall | classifier false-alarm rate |
|---|---|---|---|---|
| solar_panel | 0.666 | 0.809 | 0.923 | 0.148 |
| building | 0.898 | 0.931 | 0.947 | 0.084 |
| utility_pole | 0.996 | 0.998 | 0.996 | 0.000 |
| machinery | 0.933 | 0.945 | 0.974 | 0.117 (baseline 0.058) |

## Latency (from `reports/bench.md`, Apple M4, MPS)

| requirement | measured | target |
|---|---|---|
| NFR-1 alert latency, capture → alert, recorded video | p95 127.4 ms (p50 98.3, n = 12) | ≤ 500 ms |
| NFR-2 identity cascade @ 5,000 assets, after session-start prefetch | p95 32.3 ms (p50 9.7) | ≤ 50 ms |
| NFR-2, cold feature cache | p95 32.4 ms | ≤ 50 ms |
| NFR-3 worst start/end sync (cloud down / black-holed / slow / healthy) | 10.03 s, then offline | ≤ 10 s |

## Context-aware site risk (Chennai)

`ml/context_risk.py` scores how risky a site's surroundings are, using real public data for
the site's latitude/longitude. Missing data is reported as "n/a" and left out of the score.

| Factor | Source |
|---|---|
| Seismic zone | BIS IS 1893 (Part 1):2016, Chennai = Zone III, Z = 0.16 (`ml/regions/chennai.json`) |
| Earthquake history | USGS catalogue, largest M≥3 within 300 km since 1900 |
| Rain forecast / past 30-day rain / wind gusts | Open-Meteo (IMD thresholds: 115.6 mm/day very heavy rain, 118 km/h very severe cyclone) |
| Occupancy | OpenStreetMap tags (hospital, school, apartments …), or admin entry |
| Building age | Admin-entered construction year, or OpenStreetMap `start_date` |

The context score K ∈ [0,1] is the weighted mean of the available factors (weights in
`config.yaml → context_risk`; expert-set, not learned). It raises asset criticality,
`C' = min(1, C·(1 + 0.5·K))`, which feeds `S = (w1·M + w2·C')·Cf`, and adds `w6·K` to
revisit priority. Data is fetched in the background when a site is created or a session
starts, cached for 6 h in the Local DB, and never fetched on the per-frame path.

```bash
python scripts/context_demo.py --place "Rajiv Gandhi Government General Hospital, Chennai"     --place "Ambattur Industrial Estate, Chennai" --image wall.jpg --image wall.jpg
```

API: `GET /api/sites/{id}/context`, admin `POST /api/admin/sites/{id}/context/refresh` and
`PUT /api/admin/sites/{id}/context` (`construction_year`, `occupancy`). The pre-flight
briefing and post-flight report show the factors and their evidence.

## Limitations

- **Re-identification of heavily damaged, low-texture surfaces.** On the recorded-video
  bench, plain concrete walls with a large synthetic crack (half the image) were
  re-identified as the same asset in only 4 of 12 cases. In the end-to-end smoke test, where
  the references were also captured from video, it was 8 of 12. Plain concrete has too few
  SIFT keypoints for geometric verification, and a large crack lowers the global embedding
  similarity below the visual-only bar. The failed cases are registered as new assets but
  still alert, because the classifier sees the crack. Tag such assets with an ArUco marker,
  which the cascade checks first. Textured assets (panels, insulators, machinery) verify
  geometrically.
- **The re-identification evidence on real objects needs your camera.** Run
  `scripts/capture_reid_set.py` and then `training/evaluate_reid.py` (see `PROGRESS.md`,
  "Run on your machine"). The simulated mission histories (`make memory`) only exercise the
  pipeline; they live in their own `site_simulated` site and are labelled as simulated in
  the UI.
- **Classifier weak spots:** machinery false alarms are higher than the baseline's (0.117 vs
  0.058; MVTec "anomaly" precision 0.63), and solar `physical_damage` F1 is 0.47 with only 67
  images in total.
- **NFR-1 was measured on a 12-asset recorded video** with full-frame detection.
  With YOLO-World on the live camera, the detector adds its own time. Run
  `python scripts/bench.py --source <camera>` to measure on your hardware.
- FAISS is opt-in only. On macOS the torch and faiss-cpu wheels load conflicting OpenMP
  runtimes.
- Escalation e-mail is sent only when the `AUDI_SMTP_*` variables are set; in-app and
  WebSocket notifications always work.
