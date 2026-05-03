from __future__ import annotations

# ruff: noqa: E501

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from cuav_range.geometric import (
    CameraIntrinsics,
    DEFAULT_DRONE_SIZE_PRIORS,
    estimate_range_interval_m,
)


CLASS_NAMES = {
    0: "friendly_quad",
    1: "unknown_uav",
    2: "bird",
    3: "airplane",
    4: "helicopter",
}


@dataclass
class CameraPose:
    lat: float = 38.8895
    lon: float = -77.0353
    heading_deg: float = 35.0
    hfov_deg: float = 62.0
    focal_length_px: float = 1800.0


@dataclass
class Detection:
    xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str


@dataclass
class Track:
    track_id: int
    xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str
    misses: int = 0
    age: int = 1
    range_m: float | None = None
    range_min_m: float | None = None
    range_max_m: float | None = None
    bearing_deg: float | None = None
    lat: float | None = None
    lon: float | None = None
    priority: float = 0.0
    asset: str | None = None


class GreedyTracker:
    def __init__(self, max_distance_px: float = 90.0, max_misses: int = 12) -> None:
        self.max_distance_px = max_distance_px
        self.max_misses = max_misses
        self._next_id = 1
        self._tracks: list[Track] = []

    def update(self, detections: list[Detection]) -> list[Track]:
        assigned_tracks: set[int] = set()
        assigned_detections: set[int] = set()
        pairs: list[tuple[float, int, int]] = []
        for track_idx, track in enumerate(self._tracks):
            tx, ty = _center(track.xyxy)
            for det_idx, det in enumerate(detections):
                dx, dy = _center(det.xyxy)
                distance = math.hypot(tx - dx, ty - dy)
                if distance <= self.max_distance_px:
                    pairs.append((distance, track_idx, det_idx))

        for _, track_idx, det_idx in sorted(pairs, key=lambda item: item[0]):
            if track_idx in assigned_tracks or det_idx in assigned_detections:
                continue
            det = detections[det_idx]
            track = self._tracks[track_idx]
            track.xyxy = det.xyxy
            track.confidence = det.confidence
            track.class_id = det.class_id
            track.class_name = det.class_name
            track.misses = 0
            track.age += 1
            assigned_tracks.add(track_idx)
            assigned_detections.add(det_idx)

        for track_idx, track in enumerate(self._tracks):
            if track_idx not in assigned_tracks:
                track.misses += 1

        for det_idx, det in enumerate(detections):
            if det_idx in assigned_detections:
                continue
            self._tracks.append(
                Track(
                    track_id=self._next_id,
                    xyxy=det.xyxy,
                    confidence=det.confidence,
                    class_id=det.class_id,
                    class_name=det.class_name,
                )
            )
            self._next_id += 1

        self._tracks = [track for track in self._tracks if track.misses <= self.max_misses]
        return list(self._tracks)


class SyntheticDetector:
    """No-hardware fallback that generates a sky scene and known moving objects."""

    def __init__(self, width: int, height: int, count: int = 6, seed: int = 7) -> None:
        self.width = width
        self.height = height
        rng = random.Random(seed)
        self.objects = []
        for idx in range(count):
            x = rng.uniform(width * 0.08, width * 0.92)
            y = rng.uniform(height * 0.14, height * 0.55)
            vx = rng.uniform(-1.8, 1.8)
            vy = rng.uniform(-0.25, 0.45)
            size = rng.uniform(8, 22)
            class_id = 1 if idx < max(3, count - 2) else 2
            self.objects.append([x, y, vx, vy, size, class_id])

    def frame(self, frame_idx: int) -> tuple[np.ndarray, list[Detection]]:
        sky = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        top = np.array([170, 198, 226], dtype=np.uint8)
        bottom = np.array([220, 230, 238], dtype=np.uint8)
        for y in range(self.height):
            alpha = y / max(1, self.height - 1)
            sky[y, :, :] = (top * (1 - alpha) + bottom * alpha).astype(np.uint8)

        cv2.putText(
            sky,
            "SIMULATED CAMERA FEED",
            (28, 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (70, 80, 90),
            2,
            cv2.LINE_AA,
        )

        detections: list[Detection] = []
        for obj in self.objects:
            obj[0] += obj[2]
            obj[1] += obj[3] + math.sin(frame_idx / 23.0 + obj[4]) * 0.18
            if obj[0] < 40 or obj[0] > self.width - 40:
                obj[2] *= -1
            if obj[1] < 50 or obj[1] > self.height * 0.72:
                obj[3] *= -1

            x, y, _, _, size, class_id = obj
            half_w = size * (1.9 if class_id == 2 else 1.0)
            half_h = size * (0.45 if class_id == 2 else 0.55)
            color = (45, 45, 45) if class_id == 1 else (90, 90, 90)
            if class_id == 2:
                points = np.array(
                    [[x - half_w, y], [x, y - half_h], [x + half_w, y], [x, y + half_h]],
                    dtype=np.int32,
                )
                cv2.polylines(sky, [points], isClosed=True, color=color, thickness=2)
            else:
                cv2.circle(sky, (int(x), int(y)), max(2, int(size * 0.35)), color, -1)
                cv2.line(sky, (int(x - size), int(y)), (int(x + size), int(y)), color, 2)
                cv2.line(sky, (int(x), int(y - size)), (int(x), int(y + size)), color, 2)

            xyxy = (x - half_w, y - half_h, x + half_w, y + half_h)
            detections.append(
                Detection(
                    xyxy=xyxy,
                    confidence=0.72 + 0.2 * math.sin(frame_idx / 17.0 + size) ** 2,
                    class_id=int(class_id),
                    class_name=CLASS_NAMES[int(class_id)],
                )
            )
        return sky, detections


def run_demo(
    output_dir: Path,
    weights: Path | None = None,
    source: Path | None = None,
    synthetic: bool = False,
    frames: int = 420,
    conf: float = 0.25,
    camera_pose: CameraPose | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    camera_pose = camera_pose or CameraPose()
    tracker = GreedyTracker()
    events: list[dict] = []
    latest_tracks: list[Track] = []

    if synthetic or source is None:
        width, height, fps = 1280, 720, 30.0
        synthetic_detector = SyntheticDetector(width, height)
        model = None
        capture = None
    else:
        from ultralytics import YOLO

        if weights is None:
            raise ValueError("--weights is required when --source is provided without --synthetic")
        model = YOLO(str(weights))
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise ValueError(f"Could not open video source: {source}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
        synthetic_detector = None

    video_path = output_dir / "annotated_demo.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {video_path}")

    frame_idx = 0
    while True:
        if synthetic_detector is not None:
            if frame_idx >= frames:
                break
            frame, detections = synthetic_detector.frame(frame_idx)
        else:
            assert capture is not None
            ok, frame = capture.read()
            if not ok:
                break
            detections = _predict_frame(model, frame, conf=conf)

        tracks = tracker.update(detections)
        _enrich_tracks(tracks, width, height, camera_pose)
        annotated = _draw_dashboard_frame(frame, tracks, camera_pose)
        writer.write(annotated)
        if frame_idx % max(1, int(fps)) == 0:
            events.append(
                {
                    "frame": frame_idx,
                    "time_s": round(frame_idx / fps, 2),
                    "tracks": [_track_json(track) for track in tracks if track.misses == 0],
                }
            )
        latest_tracks = tracks
        frame_idx += 1

    writer.release()
    if capture is not None:
        capture.release()

    summary = {
        "video": video_path.name,
        "frames": frame_idx,
        "fps": fps,
        "camera": asdict(camera_pose),
        "latest_tracks": [_track_json(track) for track in latest_tracks],
        "events": events,
        "mode": "synthetic" if synthetic_detector is not None else "model",
        "weights": str(weights) if weights else None,
        "source": str(source) if source else None,
    }
    (output_dir / "telemetry.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "index.html").write_text(_render_html(summary), encoding="utf-8")
    print(f"Wrote {video_path}")
    print(f"Wrote {output_dir / 'index.html'}")


def _predict_frame(model, frame: np.ndarray, conf: float) -> list[Detection]:
    results = model.predict(frame, conf=conf, verbose=False)
    detections: list[Detection] = []
    if not results:
        return detections
    result = results[0]
    names = result.names or CLASS_NAMES
    for box in result.boxes:
        xyxy = tuple(float(value) for value in box.xyxy[0].tolist())
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        detections.append(
            Detection(
                xyxy=xyxy,
                confidence=confidence,
                class_id=class_id,
                class_name=str(names.get(class_id, CLASS_NAMES.get(class_id, f"class_{class_id}"))),
            )
        )
    return detections


def _enrich_tracks(tracks: list[Track], width: int, height: int, camera_pose: CameraPose) -> None:
    intrinsics = CameraIntrinsics(camera_pose.focal_length_px)
    for track in tracks:
        x1, _, x2, _ = track.xyxy
        bbox_w = max(1.0, x2 - x1)
        range_interval = estimate_range_interval_m(
            bbox_w,
            intrinsics,
            candidate_priors=list(DEFAULT_DRONE_SIZE_PRIORS.values()),
        )
        if range_interval:
            track.range_min_m, track.range_max_m = range_interval
            track.range_m = (track.range_min_m + track.range_max_m) / 2
        cx, _ = _center(track.xyxy)
        horizontal_offset = (cx / width - 0.5) * camera_pose.hfov_deg
        track.bearing_deg = (camera_pose.heading_deg + horizontal_offset) % 360
        if track.range_m is not None:
            track.lat, track.lon = _offset_latlon(
                camera_pose.lat,
                camera_pose.lon,
                track.bearing_deg,
                min(track.range_m, 3000.0),
            )
        center_bias = 1.0 - min(1.0, abs(cx / width - 0.5) * 2.0)
        range_score = 0.5 if track.range_m is None else max(0.0, 1.0 - min(track.range_m, 3000.0) / 3000.0)
        class_score = 0.9 if "uav" in track.class_name or "drone" in track.class_name else 0.35
        track.priority = round(100 * (0.45 * track.confidence + 0.35 * range_score + 0.20 * center_bias) * class_score, 1)

    active = sorted([track for track in tracks if track.misses == 0], key=lambda item: item.priority, reverse=True)
    assets = ["Observer A", "Observer B", "Observer C", "Observer D"]
    for idx, track in enumerate(active):
        track.asset = assets[idx] if idx < len(assets) else "Operator queue"


def _draw_dashboard_frame(frame: np.ndarray, tracks: list[Track], camera_pose: CameraPose) -> np.ndarray:
    annotated = frame.copy()
    height, width = annotated.shape[:2]
    for track in tracks:
        if track.misses:
            continue
        x1, y1, x2, y2 = [int(round(value)) for value in track.xyxy]
        color = (42, 220, 160) if track.class_id == 1 else (90, 180, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"ID {track.track_id} {track.class_name} {track.confidence:.2f}"
        if track.range_m:
            label += f" ~{track.range_m:.0f}m"
        cv2.putText(
            annotated,
            label,
            (x1, max(22, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    panel_w = 360
    overlay = annotated.copy()
    cv2.rectangle(overlay, (width - panel_w, 0), (width, height), (18, 24, 30), -1)
    annotated = cv2.addWeighted(overlay, 0.72, annotated, 0.28, 0)
    x = width - panel_w + 18
    y = 34
    _put(annotated, "DTI OPTICAL TRACKING DEMO", x, y, scale=0.58, color=(235, 240, 245))
    y += 28
    _put(annotated, "simulated geospatial projection", x, y, scale=0.45, color=(165, 175, 185))
    y += 34
    _put(annotated, f"Camera {camera_pose.lat:.4f}, {camera_pose.lon:.4f}", x, y)
    y += 22
    _put(annotated, f"Heading {camera_pose.heading_deg:.0f} deg | HFOV {camera_pose.hfov_deg:.0f} deg", x, y)
    y += 34
    active = sorted([track for track in tracks if track.misses == 0], key=lambda item: item.priority, reverse=True)
    for track in active[:8]:
        range_text = "range n/a" if track.range_m is None else f"{track.range_m:.0f}m"
        bearing_text = "brg n/a" if track.bearing_deg is None else f"{track.bearing_deg:.0f} deg"
        _put(annotated, f"#{track.track_id:02d} {track.class_name}", x, y, color=(230, 238, 245))
        y += 20
        _put(
            annotated,
            f"conf {track.confidence:.2f} | {range_text} | {bearing_text}",
            x + 8,
            y,
            scale=0.43,
            color=(178, 190, 202),
        )
        y += 19
        _put(
            annotated,
            f"priority {track.priority:.1f} | {track.asset or 'queue'}",
            x + 8,
            y,
            scale=0.43,
            color=(130, 220, 180),
        )
        y += 28
    return annotated


def _render_html(summary: dict) -> str:
    tracks_json = json.dumps(summary["latest_tracks"], indent=2)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DTI Optical Tracking Demo</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #101418;
      --panel: #171d23;
      --line: #2a343d;
      --text: #edf2f6;
      --muted: #a7b2bc;
      --accent: #45d29a;
      --warn: #f4c66a;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }}
    header {{
      height: 58px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-bottom: 1px solid var(--line);
      padding: 0 20px;
    }}
    main {{
      display: grid;
      grid-template-columns: minmax(520px, 1fr) 390px;
      min-height: calc(100vh - 58px);
    }}
    video {{
      width: 100%;
      height: calc(100vh - 58px);
      object-fit: contain;
      background: #050607;
      display: block;
    }}
    aside {{
      border-left: 1px solid var(--line);
      background: var(--panel);
      padding: 18px;
      overflow: auto;
    }}
    h1 {{ font-size: 18px; margin: 0; font-weight: 700; }}
    h2 {{ font-size: 13px; margin: 20px 0 10px; color: var(--muted); text-transform: uppercase; }}
    .status {{ color: var(--accent); font-size: 13px; }}
    .metric-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
    .metric {{ border: 1px solid var(--line); padding: 10px; border-radius: 6px; background: #12181e; }}
    .metric b {{ display: block; font-size: 20px; }}
    .metric span {{ color: var(--muted); font-size: 12px; }}
    .track {{ border-top: 1px solid var(--line); padding: 12px 0; }}
    .track strong {{ display: block; }}
    .track code {{ color: var(--accent); }}
    .map {{
      height: 270px;
      border: 1px solid var(--line);
      border-radius: 6px;
      position: relative;
      background:
        linear-gradient(rgba(255,255,255,.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,.04) 1px, transparent 1px),
        #0d1216;
      background-size: 32px 32px;
      overflow: hidden;
    }}
    .camera, .target {{
      position: absolute;
      transform: translate(-50%, -50%);
      border-radius: 999px;
    }}
    .camera {{ left: 50%; top: 70%; width: 14px; height: 14px; background: var(--warn); }}
    .target {{ width: 12px; height: 12px; background: var(--accent); box-shadow: 0 0 0 5px rgba(69,210,154,.13); }}
    footer {{ margin-top: 18px; color: var(--muted); font-size: 12px; line-height: 1.45; }}
    @media (max-width: 980px) {{
      main {{ grid-template-columns: 1fr; }}
      video {{ height: auto; }}
      aside {{ border-left: 0; border-top: 1px solid var(--line); }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>DTI Optical Tracking Demo</h1>
    <div class="status">{summary["mode"]} mode | {summary["frames"]} frames</div>
  </header>
  <main>
    <section>
      <video controls autoplay muted loop src="{summary["video"]}"></video>
    </section>
    <aside>
      <div class="metric-grid">
        <div class="metric"><b>{len(summary["latest_tracks"])}</b><span>active/recent tracks</span></div>
        <div class="metric"><b>{summary["fps"]:.1f}</b><span>source fps</span></div>
      </div>
      <h2>Camera Pose</h2>
      <div class="metric"><b>{summary["camera"]["heading_deg"]:.0f} deg</b><span>{summary["camera"]["lat"]:.5f}, {summary["camera"]["lon"]:.5f}</span></div>
      <h2>Projected Track Map</h2>
      <div class="map">
        <div class="camera" title="camera"></div>
        {_map_targets(summary["latest_tracks"])}
      </div>
      <h2>Tracks</h2>
      {_track_cards(summary["latest_tracks"])}
      <footer>
        Location dots are a demo projection from camera bearing and class-size range priors.
        They are intended for operator visualization and validation, not precise navigation.
      </footer>
    </aside>
  </main>
  <script type="application/json" id="tracks">{tracks_json}</script>
</body>
</html>
"""


def _map_targets(tracks: list[dict]) -> str:
    if not tracks:
        return ""
    pieces = []
    for track in tracks[:12]:
        bearing = float(track.get("bearing_deg") or 0)
        range_m = min(float(track.get("range_m") or 1200), 3000)
        angle = math.radians(bearing)
        distance = 0.12 + 0.48 * (range_m / 3000)
        left = 50 + math.sin(angle) * distance * 100
        top = 70 - math.cos(angle) * distance * 100
        left = max(5, min(95, left))
        top = max(5, min(95, top))
        pieces.append(
            f'<div class="target" style="left:{left:.1f}%;top:{top:.1f}%;" '
            f'title="track {track["track_id"]}"></div>'
        )
    return "\n        ".join(pieces)


def _track_cards(tracks: list[dict]) -> str:
    if not tracks:
        return '<div class="track">No tracks in final frame.</div>'
    cards = []
    for track in sorted(tracks, key=lambda item: item.get("priority", 0), reverse=True)[:10]:
        cards.append(
            f"""<div class="track">
  <strong>Track {track["track_id"]:02d} - {track["class_name"]}</strong>
  <div>confidence {track["confidence"]:.2f} - priority <code>{track["priority"]:.1f}</code></div>
  <div>range {track.get("range_m") or 0:.0f} m - bearing {track.get("bearing_deg") or 0:.0f} deg</div>
  <div>{track.get("asset") or "Operator queue"}</div>
</div>"""
        )
    return "\n".join(cards)


def _track_json(track: Track) -> dict:
    return {
        "track_id": track.track_id,
        "class_name": track.class_name,
        "confidence": round(track.confidence, 3),
        "range_m": None if track.range_m is None else round(track.range_m, 1),
        "range_min_m": None if track.range_min_m is None else round(track.range_min_m, 1),
        "range_max_m": None if track.range_max_m is None else round(track.range_max_m, 1),
        "bearing_deg": None if track.bearing_deg is None else round(track.bearing_deg, 1),
        "lat": track.lat,
        "lon": track.lon,
        "priority": track.priority,
        "asset": track.asset,
    }


def _center(xyxy: Iterable[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return (x1 + x2) / 2, (y1 + y2) / 2


def _offset_latlon(lat: float, lon: float, bearing_deg: float, distance_m: float) -> tuple[float, float]:
    radius_m = 6_371_000.0
    bearing = math.radians(bearing_deg)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    angular_distance = distance_m / radius_m
    lat2 = math.asin(
        math.sin(lat1) * math.cos(angular_distance)
        + math.cos(lat1) * math.sin(angular_distance) * math.cos(bearing)
    )
    lon2 = lon1 + math.atan2(
        math.sin(bearing) * math.sin(angular_distance) * math.cos(lat1),
        math.cos(angular_distance) - math.sin(lat1) * math.sin(lat2),
    )
    return round(math.degrees(lat2), 7), round(math.degrees(lon2), 7)


def _put(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    scale: float = 0.46,
    color: tuple[int, int, int] = (205, 215, 224),
) -> None:
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a local DTI demo video and dashboard.")
    parser.add_argument("--weights", type=Path, help="YOLO .pt checkpoint to use for real video inference.")
    parser.add_argument("--source", type=Path, help="Input video path.")
    parser.add_argument("--out", type=Path, default=Path("demo_runs/latest"))
    parser.add_argument("--synthetic", action="store_true", help="Generate a no-hardware synthetic sky demo.")
    parser.add_argument("--frames", type=int, default=420)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--camera-lat", type=float, default=CameraPose.lat)
    parser.add_argument("--camera-lon", type=float, default=CameraPose.lon)
    parser.add_argument("--heading-deg", type=float, default=CameraPose.heading_deg)
    parser.add_argument("--hfov-deg", type=float, default=CameraPose.hfov_deg)
    parser.add_argument("--focal-length-px", type=float, default=CameraPose.focal_length_px)
    args = parser.parse_args()

    pose = CameraPose(
        lat=args.camera_lat,
        lon=args.camera_lon,
        heading_deg=args.heading_deg,
        hfov_deg=args.hfov_deg,
        focal_length_px=args.focal_length_px,
    )
    run_demo(
        output_dir=args.out,
        weights=args.weights,
        source=args.source,
        synthetic=args.synthetic,
        frames=args.frames,
        conf=args.conf,
        camera_pose=pose,
    )


if __name__ == "__main__":
    main()
