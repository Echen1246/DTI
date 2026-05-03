from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLIT_ALIASES = {"valid": "val", "validation": "val", "testing": "test", "training": "train"}


@dataclass(frozen=True)
class SourceSpec:
    name: str
    root: Path
    class_map: dict[str, str | None]
    weight: float = 1.0


def merge_yolo_datasets(
    sources: list[SourceSpec],
    out: Path,
    class_names: list[str],
    link_mode: str = "hardlink",
) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    manifest_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"classes": class_names, "sources": {}}

    for source in sources:
        data_yaml = _find_data_yaml(source.root)
        source_names = _load_names(data_yaml) if data_yaml else _infer_names(source.root)
        source_name_by_id = {idx: name for idx, name in enumerate(source_names)}
        source_counts = {"train": 0, "val": 0, "test": 0, "labels": 0, "dropped_labels": 0}

        for split, image_dir, label_dir in _iter_yolo_splits(source.root, data_yaml):
            split = _canonical_split(split)
            for image_path in tqdm(list(_iter_images(image_dir)), desc=f"Merging {source.name}:{split}"):
                rel_stem = _safe_rel_stem(image_path, source.root)
                dest_image = out / "images" / split / f"{source.name}__{rel_stem}{image_path.suffix.lower()}"
                dest_label = out / "labels" / split / f"{dest_image.stem}.txt"
                _place_file(image_path, dest_image, link_mode)

                label_path = label_dir / f"{image_path.stem}.txt"
                labels, dropped = _remap_labels(label_path, source_name_by_id, source.class_map, class_to_id)
                dest_label.write_text("".join(labels), encoding="utf-8")

                source_counts[split] += 1
                source_counts["labels"] += len(labels)
                source_counts["dropped_labels"] += dropped
                manifest_rows.append(
                    {
                        "image": str(dest_image.relative_to(out)),
                        "label": str(dest_label.relative_to(out)),
                        "split": split,
                        "source": source.name,
                        "weight": source.weight,
                    }
                )

        summary["sources"][source.name] = source_counts

    with (out / "data.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "path": str(out.resolve()),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": {i: name for i, name in enumerate(class_names)},
            },
            f,
            sort_keys=False,
        )
    with (out / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "label", "split", "source", "weight"])
        writer.writeheader()
        writer.writerows(manifest_rows)
    with (out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return out / "data.yaml"


def _find_data_yaml(root: Path) -> Path | None:
    for name in ("data.yaml", "dataset.yaml", "data.yml", "dataset.yml"):
        path = root / name
        if path.exists():
            return path
    yaml_paths = list(root.rglob("data.yaml")) + list(root.rglob("dataset.yaml"))
    return yaml_paths[0] if yaml_paths else None


def _load_names(data_yaml: Path) -> list[str]:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = data.get("names", ["drone"])
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names, key=lambda x: int(x) if str(x).isdigit() else str(x))]
    return [str(name) for name in names]


def _infer_names(root: Path) -> list[str]:
    labels = sorted(root.rglob("labels/**/*.txt"))
    max_id = -1
    for label in labels[:5000]:
        for line in label.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split()
            if parts:
                try:
                    max_id = max(max_id, int(float(parts[0])))
                except ValueError:
                    pass
    if max_id == 0:
        return ["drone"]
    return [f"class_{i}" for i in range(max_id + 1)]


def _iter_yolo_splits(root: Path, data_yaml: Path | None):
    if data_yaml:
        data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
        base = Path(data.get("path") or data_yaml.parent)
        if not base.is_absolute():
            base = data_yaml.parent / base
        for key in ("train", "val", "valid", "test"):
            value = data.get(key)
            if not value:
                continue
            image_dir = _resolve_split_image_dir(data_yaml.parent, base, str(value), key)
            if image_dir.is_file():
                continue
            label_dir = _label_dir_for(image_dir)
            yield key, image_dir, label_dir
        return

    for split_dir in root.iterdir():
        if not split_dir.is_dir():
            continue
        split = _canonical_split(split_dir.name)
        image_dir = split_dir / "images"
        label_dir = split_dir / "labels"
        if image_dir.exists() and label_dir.exists():
            yield split, image_dir, label_dir


def _label_dir_for(image_dir: Path) -> Path:
    parts = list(image_dir.parts)
    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        return Path(*parts)
    return image_dir.parent / "labels"


def _resolve_split_image_dir(data_yaml_dir: Path, base: Path, value: str, split_key: str) -> Path:
    image_dir = base / value
    if image_dir.exists():
        return image_dir
    value_path = Path(value)
    split_names = [split_key, _canonical_split(split_key)]
    if split_key == "val":
        split_names.extend(["valid", "validation"])
    for split in split_names:
        candidate = data_yaml_dir / split / "images"
        if candidate.exists():
            return candidate
    if len(value_path.parts) >= 2:
        candidate = data_yaml_dir / value_path.parts[-2] / value_path.parts[-1]
        if candidate.exists():
            return candidate
    return image_dir


def _iter_images(root: Path):
    if not root.exists():
        return
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def _remap_labels(
    label_path: Path,
    source_name_by_id: dict[int, str],
    class_map: dict[str, str | None],
    class_to_id: dict[str, int],
) -> tuple[list[str], int]:
    if not label_path.exists():
        return [], 0
    labels: list[str] = []
    dropped = 0
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            source_id = int(float(parts[0]))
            x, y, w, h = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
        except ValueError:
            dropped += 1
            continue
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            dropped += 1
            continue
        source_name = source_name_by_id.get(source_id, f"class_{source_id}")
        target_name = class_map.get(source_name, class_map.get(source_name.lower()))
        if target_name is None:
            dropped += 1
            continue
        if target_name not in class_to_id:
            dropped += 1
            continue
        labels.append(f"{class_to_id[target_name]} {x:.8f} {y:.8f} {w:.8f} {h:.8f}\n")
    return labels, dropped


def _place_file(source: Path, dest: Path, link_mode: str) -> None:
    if dest.exists():
        return
    if link_mode == "symlink":
        dest.symlink_to(source)
        return
    if link_mode == "hardlink":
        try:
            dest.hardlink_to(source)
            return
        except OSError:
            pass
    shutil.copy2(source, dest)


def _canonical_split(split: str) -> str:
    return SPLIT_ALIASES.get(split.lower(), split.lower() if split.lower() in {"train", "val", "test"} else "train")


def _safe_rel_stem(path: Path, root: Path) -> str:
    try:
        return "__".join(path.relative_to(root).with_suffix("").parts)
    except ValueError:
        return path.stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge YOLO datasets into one canonical class space.")
    parser.add_argument("--sources", type=Path, required=True, help="JSON source spec list.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--classes", nargs="+", default=["friendly_quad", "unknown_uav", "bird", "airplane", "helicopter"])
    parser.add_argument("--link-mode", choices=["hardlink", "symlink", "copy"], default="hardlink")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_specs = json.loads(args.sources.read_text(encoding="utf-8"))
    sources = [
        SourceSpec(
            name=spec["name"],
            root=Path(spec["root"]).expanduser(),
            class_map=spec["class_map"],
            weight=float(spec.get("weight", 1.0)),
        )
        for spec in raw_specs
    ]
    print(merge_yolo_datasets(sources, args.out, args.classes, args.link_mode))


if __name__ == "__main__":
    main()
