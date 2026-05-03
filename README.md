# Counter-UAS Drone Detection Scaffold

This repo starts the training side of the camera-cued multi-drone detection stack:

1. Build an open combined dataset from DUT Anti-UAV, Anti-UAV, Drone-vs-Bird, and Roboflow.
2. Train a YOLO11x variant with an added P2 stride-4 detection head on Modal.
3. Persist datasets and runs in Modal Volumes so local inference/tracking can consume trained weights later.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
modal setup
```

LRDDv2 is gated by Drexel access approval/export-control clearance, so the practical first path is now the open dataset plan in [docs/open_dataset_plan.md](/Users/eddie/Documents/natsec/docs/open_dataset_plan.md).

Range estimation is handled geometrically from camera intrinsics, bbox pixel size, and drone-size priors. See [docs/range_estimation.md](/Users/eddie/Documents/natsec/docs/range_estimation.md).

## Open Combined Dataset

Canonical classes:

```yaml
0: friendly_quad
1: unknown_uav
2: bird
3: airplane
4: helicopter
```

Source manifest: [configs/datasets/open_cuas_sources.yaml](/Users/eddie/Documents/natsec/configs/datasets/open_cuas_sources.yaml)

YOLO merge helper for already-YOLO exports:

```bash
python -m cuav_data.yolo_merge \
  --sources data/source_specs/yolo_sources.json \
  --out data/datasets/open-cuas
```

Full local preparation script:

```bash
PYTHONPATH=src python scripts/prepare_open_cuas.py \
  --raw-root data/raw \
  --work-root data/datasets/sources \
  --out data/datasets/open-cuas
```

Example `data/source_specs/yolo_sources.json`:

```json
[
  {
    "name": "drone_vs_bird",
    "root": "data/raw/drone-vs-bird",
    "class_map": {"drone": "drone", "bird": "bird"}
  },
  {
    "name": "roboflow_drones_yolo11_a",
    "root": "data/raw/roboflow-drones-yolo11-a",
    "class_map": {"drone": "drone", "bird": "bird", "airplane": "airplane", "object": null, "0": null}
  }
]
```

## LRDDv2 Ingestion

Local dry run:

```bash
python -m cuav_data.lrddv2 \
  --raw /path/to/LRDDv2_or_archive \
  --out data/datasets/lrddv2 \
  --dataset-name lrddv2
```

Upload authorized archive to Modal and ingest there:

```bash
modal run modal_app/train_yolo.py --action upload --local-path /path/to/LRDDv2.zip
modal run modal_app/train_yolo.py --action ingest --remote-archive raw/lrddv2/LRDDv2.zip
```

## Start Modal Training

```bash
modal run modal_app/train_yolo.py \
  --action train \
  --data-yaml /data/datasets/open-cuas/data.yaml \
  --run-name open-cuas-yolo11x-p2-1536 \
  --epochs 200 \
  --imgsz 1536
```

The default training job requests `B200:8`, mounts:

- `cuas-data` at `/data` for raw and normalized datasets.
- `cuas-runs` at `/runs` for checkpoints and logs.

Training uses [configs/models/yolo11x-p2.yaml](/Users/eddie/Documents/natsec/configs/models/yolo11x-p2.yaml), initialized from `yolo11x.pt` through Ultralytics' `pretrained=` path.

## Current Scope

This scaffold is intentionally focused on build order step 1. Local SAHI inference, ByteTrack visualization, range estimation, and Hungarian allocation come next once the first training job is underway.

## Local Demo Platform

The demo platform can be built before the final Modal checkpoint is ready. It writes:

- `annotated_demo.mp4`: detection/tracking overlay video.
- `telemetry.json`: track, confidence, range, bearing, and simulated map metadata.
- `index.html`: self-contained dashboard for submission/screenshare.

No-hardware smoke test:

```bash
PYTHONPATH=src python -m dti_demo.platform \
  --synthetic \
  --frames 420 \
  --out demo_runs/synthetic
```

Real video once a checkpoint is downloaded:

```bash
PYTHONPATH=src python -m dti_demo.platform \
  --weights weights/best.pt \
  --source /path/to/validation_video.mp4 \
  --out demo_runs/open-cuas-demo \
  --camera-lat 38.8895 \
  --camera-lon -77.0353 \
  --heading-deg 35 \
  --hfov-deg 62
```

The map panel is a visualization layer: it projects each track from camera pose, horizontal bearing, and class-size geometric range priors. It is meant for demo/operator context, not precise navigation.

## Tabletop Terminal Frontend

For judge walkthroughs, open the static terminal UI directly:

```bash
open frontend/terminal/index.html
```

The terminal shows the intended table layout:

- Main camera feed playing the prepared annotated demo video.
- Camera-relative XYZ track space with blips and projected motion trails.
- Response asset readiness with simple track-to-asset assignments.
- Track table and operator event log.

The page is self-contained for tabletop playback: `frontend/terminal/media/real_validation_multi_target.mp4` is bundled beside the static HTML/CSS/JS. The center feed can later be swapped to another annotated MP4 or live inference stream while preserving the same right-side track-space and asset panels.
