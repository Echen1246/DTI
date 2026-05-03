from __future__ import annotations

import argparse
import json
from pathlib import Path

from cuav_data.anti_uav import prepare_anti_uav
from cuav_data.detection_convert import prepare_detection_dataset
from cuav_data.validate_yolo import validate_yolo_dataset
from cuav_data.yolo_merge import SourceSpec, merge_yolo_datasets

CLASS_NAMES = ["friendly_quad", "unknown_uav", "bird", "airplane", "helicopter"]


def main() -> None:
    args = parse_args()
    raw = args.raw_root.expanduser().resolve()
    work = args.work_root.expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)

    source_specs: list[SourceSpec] = []

    dut_root = raw / "dut-anti-uav"
    if dut_root.exists() and not args.skip_dut:
        dut_out = work / "dut-anti-uav"
        prepare_detection_dataset(
            raw=dut_root,
            out=dut_out,
            source_name="dut_anti_uav",
            default_class="unknown_uav",
        )
        source_specs.append(
            SourceSpec(
                name="dut_anti_uav",
                root=dut_out,
                class_map={"unknown_uav": "unknown_uav"},
                weight=1.5,
            )
        )

    anti_uav_root = raw / "anti-uav300"
    if anti_uav_root.exists() and not args.skip_anti_uav:
        anti_out = work / "anti-uav300"
        prepare_anti_uav(
            raw=anti_uav_root,
            out=anti_out,
            frame_stride=args.video_frame_stride,
            max_sequences=args.max_sequences,
        )
        source_specs.append(
            SourceSpec(
                name="anti_uav300",
                root=anti_out,
                class_map={"unknown_uav": "unknown_uav"},
                weight=1.25,
            )
        )

    yolo_sources_json = raw / "yolo_sources.json"
    if yolo_sources_json.exists():
        source_specs.extend(_read_yolo_sources(yolo_sources_json))

    if not source_specs:
        raise SystemExit(f"No sources found under {raw}")

    data_yaml = merge_yolo_datasets(
        sources=source_specs,
        out=args.out.expanduser().resolve(),
        class_names=CLASS_NAMES,
        link_mode=args.link_mode,
    )
    validation = validate_yolo_dataset(data_yaml, fail_on_error=args.fail_on_error)
    validation_path = data_yaml.parent / "validation_summary.json"
    validation_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print(data_yaml)
    print(validation_path)


def _read_yolo_sources(path: Path) -> list[SourceSpec]:
    specs = json.loads(path.read_text(encoding="utf-8"))
    return [
        SourceSpec(
            name=spec["name"],
            root=Path(spec["root"]).expanduser(),
            class_map=spec["class_map"],
            weight=float(spec.get("weight", 1.0)),
        )
        for spec in specs
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the combined open counter-UAS YOLO dataset.")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--work-root", type=Path, default=Path("data/datasets/sources"))
    parser.add_argument("--out", type=Path, default=Path("data/datasets/open-cuas"))
    parser.add_argument("--video-frame-stride", type=int, default=5)
    parser.add_argument("--max-sequences", type=int)
    parser.add_argument("--link-mode", choices=["hardlink", "symlink", "copy"], default="hardlink")
    parser.add_argument("--skip-dut", action="store_true")
    parser.add_argument("--skip-anti-uav", action="store_true")
    parser.add_argument("--fail-on-error", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
