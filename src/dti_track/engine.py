from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CameraModel:
    width_px: int
    height_px: int
    fx: float
    fy: float
    cx: float
    cy: float
    heading_deg: float = 0.0


@dataclass
class FilterState:
    x: float
    y: float
    z: float
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    last_time_s: float | None = None
    misses: int = 0


class OpticalTrackEngine:
    """Estimate camera-frame XYZ from single-camera detections.

    This is not true depth recovery. It combines a calibrated image ray with a
    range prior from object apparent size, then smooths the resulting state.
    """

    def __init__(
        self,
        camera: CameraModel,
        alpha: float = 0.34,
        beta: float = 0.045,
        pixel_sigma: float = 2.5,
    ) -> None:
        self.camera = camera
        self.alpha = alpha
        self.beta = beta
        self.pixel_sigma = pixel_sigma
        self._states: dict[int, FilterState] = {}

    def update_frame(self, frame: dict[str, Any]) -> dict[str, Any]:
        time_s = float(frame.get("time_s", 0.0))
        seen_ids: set[int] = set()
        states: list[dict[str, Any]] = []

        for track in frame.get("tracks", []):
            track_id = int(track["track_id"])
            measurement = self._measurement_xyz(track)
            state = self._update_track(track_id, measurement, time_s)
            seen_ids.add(track_id)
            states.append(self._serialize_state(track, measurement, state, time_s))

        for track_id, state in self._states.items():
            if track_id not in seen_ids:
                state.misses += 1

        self._states = {track_id: state for track_id, state in self._states.items() if state.misses <= 30}
        return {
            "frame": int(frame.get("frame", 0)),
            "time_s": time_s,
            "states": sorted(states, key=lambda item: item.get("priority", 0), reverse=True),
        }

    def _measurement_xyz(self, track: dict[str, Any]) -> dict[str, float]:
        x1, y1, x2, y2 = [float(value) for value in track.get("xyxy", [0, 0, 0, 0])]
        u = (x1 + x2) / 2
        v = (y1 + y2) / 2
        z = float(track.get("range_m") or 0.0)
        if z <= 0:
            z = 150.0

        xn = (u - self.camera.cx) / self.camera.fx
        yn = -(v - self.camera.cy) / self.camera.fy
        x = xn * z
        y = yn * z
        slant_range = math.sqrt(x * x + y * y + z * z)
        bearing_deg = self.camera.heading_deg + math.degrees(math.atan2(x, z))
        elevation_deg = math.degrees(math.atan2(y, math.hypot(x, z)))
        return {
            "x": x,
            "y": y,
            "z": z,
            "slant_range_m": slant_range,
            "bearing_deg": bearing_deg,
            "elevation_deg": elevation_deg,
            "pixel_u": u,
            "pixel_v": v,
            "range_min_m": float(track.get("range_min_m") or z * 0.55),
            "range_max_m": float(track.get("range_max_m") or z * 1.45),
        }

    def _update_track(self, track_id: int, measurement: dict[str, float], time_s: float) -> FilterState:
        state = self._states.get(track_id)
        if state is None:
            state = FilterState(measurement["x"], measurement["y"], measurement["z"], last_time_s=time_s)
            self._states[track_id] = state
            return state

        dt = max(1 / 30, time_s - state.last_time_s) if state.last_time_s is not None else 1 / 30
        pred_x = state.x + state.vx * dt
        pred_y = state.y + state.vy * dt
        pred_z = state.z + state.vz * dt

        rx = measurement["x"] - pred_x
        ry = measurement["y"] - pred_y
        rz = measurement["z"] - pred_z
        state.x = pred_x + self.alpha * rx
        state.y = pred_y + self.alpha * ry
        state.z = pred_z + self.alpha * rz
        state.vx += (self.beta / dt) * rx
        state.vy += (self.beta / dt) * ry
        state.vz += (self.beta / dt) * rz
        state.last_time_s = time_s
        state.misses = 0
        return state

    def _serialize_state(
        self,
        track: dict[str, Any],
        measurement: dict[str, float],
        state: FilterState,
        time_s: float,
    ) -> dict[str, Any]:
        range_sigma = max(8.0, (measurement["range_max_m"] - measurement["range_min_m"]) / 4)
        lateral_sigma = max(1.5, abs(state.z) * self.pixel_sigma / self.camera.fx)
        vertical_sigma = max(1.5, abs(state.z) * self.pixel_sigma / self.camera.fy)
        speed = math.sqrt(state.vx * state.vx + state.vy * state.vy + state.vz * state.vz)
        covariance_diag = [lateral_sigma**2, vertical_sigma**2, range_sigma**2]
        confidence = float(track.get("confidence") or 0.0)
        quality = confidence / (1.0 + range_sigma / max(1.0, state.z))
        return {
            "track_id": int(track["track_id"]),
            "class_name": track.get("class_name", "unknown_uav"),
            "confidence": round(confidence, 3),
            "priority": float(track.get("priority") or 0.0),
            "time_s": round(time_s, 4),
            "measurement_xyz_m": _round_vec([measurement["x"], measurement["y"], measurement["z"]]),
            "xyz_m": _round_vec([state.x, state.y, state.z]),
            "velocity_mps": _round_vec([state.vx, state.vy, state.vz]),
            "speed_mps": round(speed, 2),
            "bearing_deg": round(measurement["bearing_deg"], 2),
            "elevation_deg": round(measurement["elevation_deg"], 2),
            "slant_range_m": round(measurement["slant_range_m"], 2),
            "uncertainty_m": {
                "lateral_sigma": round(lateral_sigma, 2),
                "vertical_sigma": round(vertical_sigma, 2),
                "range_sigma": round(range_sigma, 2),
            },
            "covariance_diag": [round(value, 3) for value in covariance_diag],
            "quality": round(quality, 3),
            "source": "monocular_range_prior",
        }


def enrich_telemetry(telemetry: dict[str, Any], image_width: int = 1280, image_height: int = 720) -> dict[str, Any]:
    camera_info = telemetry.get("camera", {})
    focal = float(camera_info.get("focal_length_px") or 1800.0)
    camera = CameraModel(
        width_px=image_width,
        height_px=image_height,
        fx=focal,
        fy=focal,
        cx=image_width / 2,
        cy=image_height / 2,
        heading_deg=float(camera_info.get("heading_deg") or 0.0),
    )
    engine = OpticalTrackEngine(camera)
    state_frames = [engine.update_frame(frame) for frame in telemetry.get("frame_tracks", [])]
    enriched = dict(telemetry)
    enriched["image_width"] = image_width
    enriched["image_height"] = image_height
    enriched["track_state_model"] = {
        "name": "monocular_range_prior_alpha_beta",
        "coordinate_frame": "camera_xyz_m",
        "camera_model": {
            "width_px": camera.width_px,
            "height_px": camera.height_px,
            "fx": camera.fx,
            "fy": camera.fy,
            "cx": camera.cx,
            "cy": camera.cy,
            "heading_deg": camera.heading_deg,
        },
        "limitations": [
            "Depth is estimated from object-size/range priors, not directly observed.",
            "Single-camera XYZ should be displayed with uncertainty.",
            "Multi-camera triangulation or external ranging is required for precise 3D localization.",
        ],
    }
    enriched["track_state_frames"] = state_frames
    enriched["latest_track_states"] = state_frames[-1]["states"] if state_frames else []
    return enriched


def write_js(telemetry: dict[str, Any], output: Path, variable: str = "DTI_TELEMETRY") -> None:
    output.write_text(f"window.{variable} = {json.dumps(telemetry, separators=(',', ':'))};\n", encoding="utf-8")


def _round_vec(values: list[float]) -> list[float]:
    return [round(value, 3) for value in values]


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate camera-frame XYZ track state from DTI video telemetry.")
    parser.add_argument("--input", type=Path, required=True, help="Input telemetry.json from dti_demo.platform.")
    parser.add_argument("--output", type=Path, required=True, help="Output enriched telemetry JSON.")
    parser.add_argument("--js-output", type=Path, help="Optional frontend JS wrapper output.")
    parser.add_argument("--image-width", type=int, default=1280)
    parser.add_argument("--image-height", type=int, default=720)
    args = parser.parse_args()

    telemetry = json.loads(args.input.read_text(encoding="utf-8"))
    enriched = enrich_telemetry(telemetry, image_width=args.image_width, image_height=args.image_height)
    args.output.write_text(json.dumps(enriched, indent=2), encoding="utf-8")
    if args.js_output:
        write_js(enriched, args.js_output)
    print(f"Wrote {args.output}")
    if args.js_output:
        print(f"Wrote {args.js_output}")


if __name__ == "__main__":
    main()
