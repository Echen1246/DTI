# Open Dataset Plan

LRDDv2 is useful on paper but blocked by Drexel permission gating. The open-first stack should be:

1. **DUT Anti-UAV** as the primary detection source.
2. **Anti-UAV300** as the multi-frame tracking/video source.
3. **Drone-vs-Bird** as hard negatives and explicit bird discrimination.
4. **Roboflow `drones-yolo11-a`** as a fast YOLO-formatted volume booster.

Range estimation is not trained from dataset labels in the first pass. Use geometric ranging:

```text
distance_m ~= focal_length_px * real_object_size_m / bbox_size_px
```

The physical size prior comes from drone type/classification or a conservative default class, and camera intrinsics come from calibration.

## Recommendation

Use a four-class detector for the first training run:

```yaml
0: drone
1: bird
2: airplane
3: helicopter
```

This is better than a drone-only detector for the counter-UAS demo because birds/airplanes/helicopters become explicit non-engagement classes instead of unlabeled background. At inference time, only `drone` detections go into threat ranking and Hungarian allocation; the others are displayed or logged as suppressors.

## Source Notes

### DUT Anti-UAV

- Official repository: https://github.com/wangdongdut/DUT-Anti-UAV
- Provides anti-UAV detection train/val/test downloads and separate tracking downloads.
- Use for: primary detection training in the ground-camera counter-UAS viewpoint.
- Conversion: inspect the downloaded annotation format, then normalize to YOLO `drone` labels. The converter should support COCO, VOC XML, or preexisting YOLO layouts.

### Anti-UAV300

- Official repository: https://github.com/ZhaoJ9014/Anti-UAV
- The repo describes Anti-UAV as RGB and/or IR video tracking data with dense boxes and target-existence flags.
- The maintainers currently list three public datasets: Anti-UAV300, Anti-UAV410, and Anti-UAV600. Anti-UAV300 is the preferred first source because it includes RGB and IR; 410/600 are IR-only.
- Use for: main detector examples, camera-like viewing geometry, frame-to-frame continuity, ByteTrack validation.
- Conversion: sample frames from video, convert visible/existence tracking boxes to YOLO `drone`; preserve `sequence_id`, `frame_id`, and `visible` in the manifest.

### Drone-vs-Bird

- Mendeley Data v5: https://data.mendeley.com/datasets/6ghdz52pd7/5
- Data in Brief paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC10440445/
- 20,925 JPEG images at 640x640, YOLOv7-style `train`, `valid`, `test` folders with `images` and `labels`.
- License: CC BY 4.0.
- Use for: hard negatives and bird class discrimination. Keep `bird` labels instead of dropping them.

### Roboflow `drones-yolo11-a`

- Roboflow Universe: https://universe.roboflow.com/drone-a7lpy/drones-yolo11-a
- About 9.9k images, YOLO11-tagged project, classes include `bird`, `drone`, `airplane`, `object`, and `0`.
- License shown by Roboflow: BY-NC-SA 4.0.
- Use for: quick volume boost only after label sanity checks. Map `bird`, `drone`, and `airplane`; ignore ambiguous `object` and `0` until inspected.

## First-Hour Pull Plan

```bash
mkdir -p data/raw data/datasets

# DUT Anti-UAV: use the Google Drive/Baidu detection links from the official repo.
# Save/extract under data/raw/dut-anti-uav.

# Drone-vs-Bird: use the Mendeley Data page/API and save the downloaded archive here.
# https://data.mendeley.com/datasets/6ghdz52pd7/5

# Roboflow export, requires Roboflow CLI/API auth.
roboflow download -f yolov11 -l data/raw/roboflow-drones-yolo11-a drone-a7lpy/drones-yolo11-a/14
```

Anti-UAV300 uses Google Drive/Baidu links from the official repo. Download Anti-UAV300 first, not 410/600, because we want RGB for the first YOLO pass.

## Integration Order

1. Merge already-YOLO datasets first: Drone-vs-Bird + Roboflow.
2. Add DUT Anti-UAV detection data.
3. Add Anti-UAV300 RGB video-frame extraction and tracking manifest.
4. Train YOLO11x-P2 on the combined detector data.
5. Add geometric range estimation in local inference using calibrated camera intrinsics and drone-size priors.
