# DTI Project Checkpoint - 2026-05-02

DTI means Detect, Track, Intercept. Current implementation focus is the Detect data pipeline and Modal training scaffold for a passive optical counter-UAS demo.

## Current Dataset Decision

Active public datasets:

- DUT Anti-UAV: primary ground-camera detection set, Pascal VOC XML.
- Anti-UAV300: visible-camera video/tracking source, JSON annotations plus MP4; used for frame extraction and tracking validation.
- Drone-vs-Bird: YOLO hard-negative set; drone maps to `unknown_uav`, bird maps to `bird`.
- Roboflow `drones-yolo11-a`: volume booster, YOLO-formatted export downloaded through the Modal `roboflow-api` secret.

Dropped:

- LRDDv2: gated behind Drexel access.
- Halmstad: MATLAB groundTruth objects are not directly usable without MATLAB/pre-conversion.

Range estimation plan:

- No supervised range regression yet.
- Use geometric ranging from bbox size, camera intrinsics, and class/vehicle size priors.

## Class Schema

```yaml
0: friendly_quad
1: unknown_uav
2: bird
3: airplane
4: helicopter
```

Current public drone labels map to `unknown_uav`. Our own known friendly quadrotor data should be added later as `friendly_quad`.

## Modal State

Persistent volumes:

- `cuas-data`
- `cuas-runs`

Confirmed raw archives in `cuas-data`:

- `raw/dut-anti-uav/train.zip`
- `raw/dut-anti-uav/val.zip`
- `raw/dut-anti-uav/test.zip`
- `raw/anti-uav300/anti-uav300.zip`
- `raw/drone-vs-bird/drone-vs-bird.zip`
- `raw/roboflow/drones-yolo11-a.zip`

Roboflow export details:

- Workspace/project/version: `drone-a7lpy/drones-yolo11-a/16`
- Export format: YOLO
- Native classes: `0`, `airplane`, `bird`, `drone`
- DTI mapping: `0` and `drone` map to `unknown_uav`; `bird` and `airplane` are preserved.

Interrupted full prep run:

- App ID: `ap-SRcNI7wIGmvlMLAtAhJdAp`
- State after connection loss: stopped
- It had completed DUT conversion and was converting Anti-UAV sequences when the client heartbeat failed.

Known-good smoke run:

- Output: `/data/datasets/open-cuas-smoke3/data.yaml`
- Included DUT, Anti-UAV visible frames with `max_sequences=3`, and Drone-vs-Bird.
- Validation showed the merge worked, but Drone-vs-Bird had bad YOLO boxes before the fix.

Completed full prep run:

- App ID: `ap-AnBaYb5ZUOXbSntgeAOMfX`
- Output: `/data/datasets/open-cuas/data.yaml`
- Sources: DUT Anti-UAV, Anti-UAV300, Drone-vs-Bird, Roboflow `drones-yolo11-a`
- Validation: zero errors, zero bad boxes, zero missing label files.
- Final split sizes:
  - train: 51,510 images, 78,540 labels
  - val: 12,174 images, 12,442 labels
  - test: 12,094 images, 12,189 labels
- Final class counts:
  - `unknown_uav`: 94,172
  - `bird`: 4,859
  - `airplane`: 4,140
  - `friendly_quad`: 0
  - `helicopter`: 0

Source summary:

- DUT Anti-UAV: 10,000 images, 10,108 kept labels, 1 dropped invalid label.
- Anti-UAV300: 28,047 sampled visible frames/labels from 318 sequences.
- Drone-vs-Bird: 20,952 images, 46,888 kept labels, 961 dropped invalid labels.
- Roboflow `drones-yolo11-a`: 16,779 images, 18,128 kept labels, 1,447 dropped invalid labels.

## Code Changes Made

Key files:

- `modal_app/train_yolo.py`
  - Added Modal download helpers for Google Drive and generic URLs.
  - Added ZIP inspection and text-read helpers.
  - Added `prepare_open_cuas` action to extract, convert, merge, and validate datasets in Modal.
  - Added Roboflow secret-backed downloader and automatic Roboflow ZIP extraction in `prepare_open_cuas`.
  - Uses `/tmp/open_cuas_prepare` for extraction/conversion staging to avoid slow volume traversal.

- `src/cuav_data/detection_convert.py`
  - Converts VOC, COCO, and YOLO detection datasets into canonical YOLO.
  - Added class remapping to DTI schema.
  - Added VOC format hint and faster image resolution for DUT split layout.

- `src/cuav_data/anti_uav.py`
  - Converts Anti-UAV-style video/tracking JSON into YOLO frame images.
  - Defaults to visible modality.
  - Preserves train/val/test split inferred from directory path.

- `src/cuav_data/yolo_merge.py`
  - Merges YOLO sources into one class schema.
  - Resolves Roboflow-style `../train/images` paths.
  - Drops invalid/out-of-range YOLO boxes during merge.

- `src/cuav_data/validate_yolo.py`
  - Validates image/label presence, class IDs, boxes, and class counts.

## Current Blockers

- Current working scaffold is in `/Users/eddie/Documents/natsec`.
- GitHub target repo was cloned into `/Users/eddie/Documents/natsec/DTI`.
- Initial scaffold checkpoint was pushed to `main` as commit `bd44b98`.
- First YOLO11x-P2 training job was spawned after fixing multi-GPU batch sizing.
- Training app: `ap-xcc3QtepwhjtNyvTFjgUkU`
- Training function call: `fc-01KQPD527KTBQNZ0ZDGNJCDN7W`
- Training config: `epochs=200`, `imgsz=1536`, `batch=64`, `device=0,1,2,3,4,5,6,7`
- Earlier failed launch `ap-hwdZ1pcmme2MDgMdnvkYF1` died before epoch 1 because Ultralytics does not support `batch=-1` AutoBatch in multi-GPU training.

## Next Commands

Dataset prep is complete. Training was launched with:

```bash
modal run -d modal_app/train_yolo.py --action train_spawn --data-yaml /data/datasets/open-cuas/data.yaml --epochs 200 --imgsz 1536 --batch 64
```

Check prep result:

```bash
modal run modal_app/train_yolo.py --action status
modal volume ls cuas-data datasets/open-cuas
```

Once `datasets/open-cuas/data.yaml` validates cleanly, launch training:

```bash
modal run modal_app/train_yolo.py --action train --data-yaml /data/datasets/open-cuas/data.yaml --epochs 200 --imgsz 1536 --batch -1
```

Use Modal logs when a run looks quiet:

```bash
modal app list
modal app logs <app-id>
```
