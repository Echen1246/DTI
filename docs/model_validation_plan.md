# Model Validation Plan

We validate usefulness by measuring whether the system supports passive optical last-mile tracking and visual ID, not just whether YOLO scores well on static images.

## Offline Detector Metrics

- `mAP50-95` for overall detector quality.
- `AP_small` or custom small-object AP for long-range targets.
- Per-class precision/recall for:
  - `friendly_quad`
  - `unknown_uav`
  - `bird`
  - `airplane`
  - `helicopter`
- False-positive rate on bird-heavy clips.
- False-negative rate on tiny UAV frames.

## Tracking / Lock Metrics

Use Anti-UAV and any staged video clips as sequence-level validation.

- Track continuity: percent of target-present frames with a valid track.
- ID switches per minute.
- Track fragmentation count per sequence.
- Reacquisition time after missed detections.
- Lock survival through abrupt motion: hold rate after high image-plane acceleration.

## Visual IFF Metrics

For the hackathon demo, treat IFF as visual class support, not a magical allegiance oracle.

- Known friendly quad vs unknown UAV confusion matrix.
- Bird vs unknown UAV confusion matrix.
- Unknown/low-confidence rate when silhouette is too small or ambiguous.
- Human-in-loop override path for uncertain IDs.

## Range Metrics

Range is geometric in this phase.

- Calibration sanity check on known-distance footage.
- Relative range ordering accuracy: nearer targets rank nearer than farther targets.
- Range interval coverage: true range falls inside estimated interval when test footage has measured distance.

## End-To-End Demo Criteria

A model is useful if it can:

1. Detect a small UAV at long range.
2. Preserve a track ID through evasive motion or occlusion gaps.
3. Suppress bird false alarms.
4. Distinguish friendly quad-like targets from unknown UAV-like targets when visually resolvable.
5. Produce a range band and threat priority for assignment.
