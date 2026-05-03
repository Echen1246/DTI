from __future__ import annotations

from dti_track.engine import enrich_telemetry


def test_enrich_telemetry_adds_xyz_state() -> None:
    telemetry = {
        "frames": 1,
        "fps": 30,
        "camera": {"focal_length_px": 1000, "heading_deg": 10},
        "frame_tracks": [
            {
                "frame": 0,
                "time_s": 0,
                "tracks": [
                    {
                        "track_id": 1,
                        "class_name": "unknown_uav",
                        "confidence": 0.8,
                        "xyxy": [630, 350, 650, 370],
                        "range_m": 100,
                        "range_min_m": 70,
                        "range_max_m": 150,
                        "priority": 75,
                    }
                ],
            }
        ],
    }

    enriched = enrich_telemetry(telemetry, image_width=1280, image_height=720)
    state = enriched["track_state_frames"][0]["states"][0]

    assert enriched["image_width"] == 1280
    assert enriched["image_height"] == 720
    assert state["track_id"] == 1
    assert state["xyz_m"][2] == 100
    assert state["bearing_deg"] == 10
    assert state["uncertainty_m"]["range_sigma"] == 20
