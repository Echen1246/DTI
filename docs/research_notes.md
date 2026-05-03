# Training And Dataset Notes

## LRDDv2

- Official Drexel/iMAPLE page describes LRDDv2 as 39,516 annotated images for long-range drone detection, with range information for over 8,000 images.
- Access is not a public direct download; requesters submit a form and Drexel clears access under current U.S. export-control regulations before sending a link.
- Operational impact: the repo should not embed credentials or scraper logic. We ingest a user-provided authorized archive or directory and normalize it into YOLO layout.
- Source: https://research.coe.drexel.edu/ece/imaple/lrddv2/

## Anti-UAV

- Official Anti-UAV repository frames the benchmark as UAV tracking data, with visible and infrared versions depending on release.
- Anti-UAV v1 is described in the paper as 318 RGB-T video pairs with dense boxes, attributes, and target-existence flags; Anti-UAV410 is a later thermal benchmark with 410 videos.
- Operational impact: useful for later detector pretraining or tracker stress testing, but its annotations are video/tracking oriented. We should convert visible RGB frames into detection samples carefully and preserve frame ids for ByteTrack validation.
- Source: https://github.com/ZhaoJ9014/Anti-UAV

## Drones-vs-Bird

- Mendeley/Data in Brief dataset is YOLOv7-style with train/valid/test folders, images and labels, 20,925 640x640 JPEG images, and drone/bird classes.
- Operational impact: use it to teach the detector discriminative bird negatives. For a defender allocation pipeline, keep bird as a class during training or map bird to hard-negative/no-target depending on the false-positive tolerance experiment.
- Source: https://data.mendeley.com/datasets/6ghdz52pd7/5
- Paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC10440445/

## YOLO11x + P2

- Ultralytics YOLO11 detect models support training from YAML or pretrained `.pt`; the official model card lists YOLO11x as the largest detection variant at 56.9M parameters and 194.9B FLOPs at 640.
- The official YOLO11 detect YAML outputs P3/P4/P5. For LRDDv2, where drones are frequently tiny in 1080p frames, adding P2/4 improves feature resolution before SAHI-tiled inference.
- Operational impact: train `configs/models/yolo11x-p2.yaml` with `pretrained=yolo11x.pt`, accepting partial transfer for the new P2 head.
- Sources:
  - https://huggingface.co/Ultralytics/YOLO11
  - https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/11/yolo11.yaml
  - https://docs.ultralytics.com/modes/train/

## Modal

- Modal supports B200 GPUs via `gpu="B200"` and multi-GPU single-node training by appending a count, e.g. `gpu="B200:8"`.
- Modal Volumes are durable shared filesystems; writes should be committed and reads reloaded when another container has modified the volume. They are designed for write-once/read-many workloads such as datasets and model checkpoints.
- Local clients can upload into a Volume with `volume.batch_upload()`; the CLI equivalent is `modal volume put`.
- Operational impact: use `cuas-data` for authorized raw archives and normalized datasets, and `cuas-runs` for Ultralytics checkpoints/logs.
- Sources:
  - https://modal.com/docs/guide/gpu
  - https://modal.com/docs/guide/volumes
  - https://modal.com/docs/guide/secrets

## SAHI And ByteTrack

- SAHI performs sliced/tiled inference, merges detections back into full-image coordinates, and has Ultralytics integration. This is the right local demo path for small long-range drones in high-resolution frames.
- ByteTrack associates both high-score and lower-score detection boxes to reduce track fragmentation, which is useful for small intermittent drone detections.
- Operational impact: the next local demo should feed SAHI detections into a ByteTrack adapter, then visualize track ids, confidence, priority, and later assignment.
- Sources:
  - https://obss.github.io/sahi/notebooks/inference_for_ultralytics/
  - https://github.com/FoundationVision/ByteTrack
