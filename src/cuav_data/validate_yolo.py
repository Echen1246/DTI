from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def validate_yolo_dataset(data_yaml: Path, fail_on_error: bool = False) -> dict[str, Any]:
    data_yaml = data_yaml.expanduser().resolve()
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(data.get("path") or data_yaml.parent)
    if not root.is_absolute():
        root = data_yaml.parent / root
    names = data.get("names", [])
    class_count = len(names)
    if isinstance(names, dict):
        class_count = len(names.keys())

    summary: dict[str, Any] = {
        "data_yaml": str(data_yaml),
        "root": str(root),
        "class_count": class_count,
        "splits": {},
        "errors": [],
    }

    for split_name in ("train", "val", "test"):
        split_value = data.get(split_name)
        if not split_value:
            continue
        image_dir = root / str(split_value)
        label_dir = _label_dir_for(image_dir)
        split_summary = {
            "images": 0,
            "labels": 0,
            "empty_label_files": 0,
            "missing_label_files": 0,
            "bad_lines": 0,
            "bad_class_ids": 0,
            "bad_boxes": 0,
            "class_counts": {str(i): 0 for i in range(class_count)},
        }
        if not image_dir.exists():
            summary["errors"].append(f"{split_name}: image dir missing: {image_dir}")
            summary["splits"][split_name] = split_summary
            continue

        for image_path in _iter_images(image_dir):
            split_summary["images"] += 1
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                split_summary["missing_label_files"] += 1
                continue
            lines = [line.strip() for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines()]
            if not lines:
                split_summary["empty_label_files"] += 1
                continue
            for line in lines:
                parsed = _parse_label_line(line)
                if parsed is None:
                    split_summary["bad_lines"] += 1
                    continue
                class_id, x, y, w, h = parsed
                if class_id < 0 or class_id >= class_count:
                    split_summary["bad_class_ids"] += 1
                    continue
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    split_summary["bad_boxes"] += 1
                    continue
                split_summary["labels"] += 1
                split_summary["class_counts"][str(class_id)] += 1

        for key in ("missing_label_files", "bad_lines", "bad_class_ids", "bad_boxes"):
            if split_summary[key]:
                summary["errors"].append(f"{split_name}: {key}={split_summary[key]}")
        summary["splits"][split_name] = split_summary

    if fail_on_error and summary["errors"]:
        raise SystemExit(json.dumps(summary, indent=2))
    return summary


def _label_dir_for(image_dir: Path) -> Path:
    parts = list(image_dir.parts)
    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        return Path(*parts)
    return image_dir.parent / "labels"


def _iter_images(root: Path):
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def _parse_label_line(line: str) -> tuple[int, float, float, float, float] | None:
    parts = line.split()
    if len(parts) < 5:
        return None
    try:
        return int(float(parts[0])), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
    except ValueError:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate an Ultralytics YOLO dataset.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--fail-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = validate_yolo_dataset(args.data, fail_on_error=args.fail_on_error)
    text = json.dumps(summary, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
