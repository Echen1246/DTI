from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from PIL import Image
from tqdm import tqdm

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VAL_NAMES = {"val", "valid", "validation"}
TEST_NAMES = {"test", "testing"}
TRAIN_NAMES = {"train", "training"}
RANGE_KEYS = ("range", "range_m", "distance", "distance_m", "target_range", "target_range_m")


@dataclass(frozen=True)
class YoloObject:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


@dataclass(frozen=True)
class ImageRecord:
    source: Path
    split: str
    labels: tuple[YoloObject, ...]
    range_m: float | None = None


def prepare_lrddv2(
    raw: Path,
    out: Path,
    dataset_name: str = "lrddv2",
    seed: int = 42,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    link_mode: str = "hardlink",
) -> Path:
    raw_root = _materialize_raw(raw, out.parent / f".{dataset_name}_raw")
    if _looks_like_yolo(raw_root):
        records, names = _read_yolo_tree(raw_root)
    else:
        coco_files = _find_coco_jsons(raw_root)
        if coco_files:
            records, names = _read_coco(coco_files, raw_root, seed, val_fraction, test_fraction)
        else:
            records, names = _read_flat_with_sidecars(raw_root, seed, val_fraction, test_fraction)

    if not records:
        raise ValueError(f"No LRDDv2 image records found under {raw_root}")

    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    for record in tqdm(records, desc="Writing YOLO dataset"):
        split = _canonical_split(record.split)
        image_dest = out / "images" / split / _unique_name(record.source, raw_root)
        label_dest = out / "labels" / split / f"{image_dest.stem}.txt"
        _place_file(record.source, image_dest, link_mode)
        _write_label(label_dest, record.labels)
        if record.range_m is not None:
            manifest_rows.append(
                {
                    "image": str(image_dest.relative_to(out)),
                    "split": split,
                    "range_m": record.range_m,
                }
            )

    data_yaml = out / "data.yaml"
    with data_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "path": str(out.resolve()),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": {i: name for i, name in enumerate(names)},
            },
            f,
            sort_keys=False,
        )

    if manifest_rows:
        with (out / "range_manifest.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["image", "split", "range_m"])
            writer.writeheader()
            writer.writerows(manifest_rows)

    _write_summary(out, records, names)
    return data_yaml


def _materialize_raw(raw: Path, extract_dir: Path) -> Path:
    raw = raw.expanduser().resolve()
    if raw.is_dir():
        return raw
    if not raw.exists():
        raise FileNotFoundError(raw)

    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    if zipfile.is_zipfile(raw):
        with zipfile.ZipFile(raw) as zf:
            zf.extractall(extract_dir)
    elif tarfile.is_tarfile(raw):
        with tarfile.open(raw) as tf:
            tf.extractall(extract_dir)
    else:
        raise ValueError(f"Unsupported archive format: {raw}")

    children = [p for p in extract_dir.iterdir() if p.is_dir()]
    return children[0] if len(children) == 1 else extract_dir


def _looks_like_yolo(root: Path) -> bool:
    split_dirs = [root / split for split in ("train", "val", "valid", "test")]
    return any((p / "images").is_dir() and (p / "labels").is_dir() for p in split_dirs) or (
        (root / "images").is_dir() and (root / "labels").is_dir()
    )


def _read_yolo_tree(root: Path) -> tuple[list[ImageRecord], list[str]]:
    names = _read_names_from_yaml(root) or ["drone"]
    records: list[ImageRecord] = []
    split_roots = [p for p in root.iterdir() if p.is_dir() and _canonical_split(p.name) in {"train", "val", "test"}]
    if not split_roots and (root / "images").is_dir():
        split_roots = [root]

    for split_root in split_roots:
        inferred_split = _canonical_split(split_root.name if split_root != root else "train")
        image_base = split_root / "images"
        label_base = split_root / "labels"
        if split_root == root and any((image_base / s).is_dir() for s in ("train", "val", "valid", "test")):
            for split_dir in image_base.iterdir():
                split = _canonical_split(split_dir.name)
                labels_dir = label_base / split_dir.name
                records.extend(_records_from_yolo_split(split_dir, labels_dir, split))
        else:
            records.extend(_records_from_yolo_split(image_base, label_base, inferred_split))
    return records, names


def _records_from_yolo_split(image_dir: Path, label_dir: Path, split: str) -> list[ImageRecord]:
    records = []
    for image in _iter_images(image_dir):
        label_path = label_dir / f"{image.stem}.txt"
        labels = _read_yolo_label(label_path)
        records.append(ImageRecord(source=image, split=split, labels=tuple(labels)))
    return records


def _find_coco_jsons(root: Path) -> list[Path]:
    jsons = []
    for path in root.rglob("*.json"):
        try:
            with path.open(encoding="utf-8") as f:
                obj = json.load(f)
        except Exception:
            continue
        if isinstance(obj, dict) and "images" in obj and "annotations" in obj:
            jsons.append(path)
    return jsons


def _read_coco(
    coco_files: list[Path],
    root: Path,
    seed: int,
    val_fraction: float,
    test_fraction: float,
) -> tuple[list[ImageRecord], list[str]]:
    records: list[ImageRecord] = []
    names: list[str] = []
    for json_path in coco_files:
        with json_path.open(encoding="utf-8") as f:
            data = json.load(f)
        categories = sorted(data.get("categories", [{"id": 1, "name": "drone"}]), key=lambda c: c["id"])
        cat_to_idx = {cat["id"]: idx for idx, cat in enumerate(categories)}
        if len(categories) > len(names):
            names = [str(cat.get("name", f"class_{i}")) for i, cat in enumerate(categories)]

        annotations_by_image: dict[int, list[dict[str, Any]]] = {}
        for ann in data.get("annotations", []):
            annotations_by_image.setdefault(int(ann["image_id"]), []).append(ann)

        image_entries = data.get("images", [])
        split_by_id = _assign_splits(image_entries, seed, val_fraction, test_fraction)
        for image_info in image_entries:
            image_path = _resolve_image_path(root, json_path.parent, image_info.get("file_name", ""))
            width, height = _image_size(image_path, image_info)
            labels: list[YoloObject] = []
            ranges: list[float] = []
            for ann in annotations_by_image.get(int(image_info["id"]), []):
                if ann.get("iscrowd", 0):
                    continue
                bbox = ann.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue
                x, y, w, h = map(float, bbox)
                labels.append(
                    YoloObject(
                        class_id=cat_to_idx.get(ann.get("category_id"), 0),
                        x_center=(x + w / 2) / width,
                        y_center=(y + h / 2) / height,
                        width=w / width,
                        height=h / height,
                    )
                )
                maybe_range = _extract_range(ann)
                if maybe_range is not None:
                    ranges.append(maybe_range)
            image_range = _extract_range(image_info)
            if image_range is None and ranges:
                image_range = sum(ranges) / len(ranges)
            records.append(
                ImageRecord(
                    source=image_path,
                    split=split_by_id[int(image_info["id"])],
                    labels=tuple(labels),
                    range_m=image_range,
                )
            )
    return records, names or ["drone"]


def _read_flat_with_sidecars(
    root: Path,
    seed: int,
    val_fraction: float,
    test_fraction: float,
) -> tuple[list[ImageRecord], list[str]]:
    sidecar_ranges = _read_range_sidecars(root)
    images = list(_iter_images(root))
    split_by_path = _assign_splits([{"id": i, "file_name": str(p)} for i, p in enumerate(images)], seed, val_fraction, test_fraction)
    records = []
    for i, image in enumerate(images):
        labels = _read_yolo_label(image.with_suffix(".txt"))
        records.append(
            ImageRecord(
                source=image,
                split=split_by_path[i],
                labels=tuple(labels),
                range_m=sidecar_ranges.get(image.name) or sidecar_ranges.get(str(image.relative_to(root))),
            )
        )
    return records, ["drone"]


def _read_range_sidecars(root: Path) -> dict[str, float]:
    ranges: dict[str, float] = {}
    for csv_path in root.rglob("*.csv"):
        try:
            with csv_path.open(encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    continue
                range_key = next((k for k in reader.fieldnames if k.lower() in RANGE_KEYS), None)
                image_key = next((k for k in reader.fieldnames if k.lower() in {"image", "filename", "file_name", "path"}), None)
                if not range_key or not image_key:
                    continue
                for row in reader:
                    value = _to_float(row.get(range_key))
                    if value is not None and row.get(image_key):
                        ranges[row[image_key]] = value
                        ranges[Path(row[image_key]).name] = value
        except Exception:
            continue
    return ranges


def _assign_splits(
    image_entries: list[dict[str, Any]],
    seed: int,
    val_fraction: float,
    test_fraction: float,
) -> dict[int, str]:
    split_by_id = {}
    remaining = []
    for item in image_entries:
        file_name = str(item.get("file_name", "")).lower()
        explicit = next((s for s in ("train", "val", "test") if f"/{s}/" in file_name or f"\\{s}\\" in file_name), None)
        if explicit:
            split_by_id[int(item["id"])] = explicit
        else:
            remaining.append(int(item["id"]))

    rng = random.Random(seed)
    rng.shuffle(remaining)
    n = len(remaining)
    n_test = int(n * test_fraction)
    n_val = int(n * val_fraction)
    for image_id in remaining[:n_test]:
        split_by_id[image_id] = "test"
    for image_id in remaining[n_test : n_test + n_val]:
        split_by_id[image_id] = "val"
    for image_id in remaining[n_test + n_val :]:
        split_by_id[image_id] = "train"
    return split_by_id


def _canonical_split(split: str) -> str:
    lowered = split.lower()
    if lowered in VAL_NAMES:
        return "val"
    if lowered in TEST_NAMES:
        return "test"
    if lowered in TRAIN_NAMES:
        return "train"
    return "train"


def _iter_images(root: Path):
    if not root.exists():
        return
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def _read_yolo_label(path: Path) -> list[YoloObject]:
    if not path.exists():
        return []
    labels = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id, x, y, w, h = parts[:5]
            labels.append(YoloObject(int(float(class_id)), float(x), float(y), float(w), float(h)))
    return labels


def _write_label(path: Path, labels: tuple[YoloObject, ...]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for label in labels:
            f.write(
                f"{label.class_id} "
                f"{_clip01(label.x_center):.8f} {_clip01(label.y_center):.8f} "
                f"{_clip01(label.width):.8f} {_clip01(label.height):.8f}\n"
            )


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _read_names_from_yaml(root: Path) -> list[str] | None:
    for yaml_path in list(root.glob("*.yaml")) + list(root.glob("*.yml")):
        try:
            with yaml_path.open(encoding="utf-8") as f:
                data = yaml.safe_load(f)
            names = data.get("names") if isinstance(data, dict) else None
        except Exception:
            continue
        if isinstance(names, dict):
            return [str(names[k]) for k in sorted(names)]
        if isinstance(names, list):
            return [str(name) for name in names]
    return None


def _resolve_image_path(root: Path, annotation_dir: Path, file_name: str) -> Path:
    candidates = [
        root / file_name,
        annotation_dir / file_name,
        root / "images" / file_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = list(root.rglob(Path(file_name).name))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not resolve image for COCO file_name={file_name}")


def _image_size(path: Path, image_info: dict[str, Any]) -> tuple[float, float]:
    width = image_info.get("width")
    height = image_info.get("height")
    if width and height:
        return float(width), float(height)
    with Image.open(path) as img:
        return float(img.width), float(img.height)


def _extract_range(item: dict[str, Any]) -> float | None:
    for key in RANGE_KEYS:
        if key in item:
            value = _to_float(item[key])
            if value is not None:
                return value
    attrs = item.get("attributes")
    if isinstance(attrs, dict):
        for key in RANGE_KEYS:
            if key in attrs:
                value = _to_float(attrs[key])
                if value is not None:
                    return value
    return None


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _unique_name(path: Path, root: Path) -> str:
    try:
        rel = path.relative_to(root)
        stem = "__".join(rel.with_suffix("").parts)
        return f"{stem}{path.suffix.lower()}"
    except ValueError:
        return path.name


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


def _write_summary(out: Path, records: list[ImageRecord], names: list[str]) -> None:
    counts = {"train": 0, "val": 0, "test": 0}
    labeled = {"train": 0, "val": 0, "test": 0}
    range_count = 0
    for record in records:
        split = _canonical_split(record.split)
        counts[split] += 1
        if record.labels:
            labeled[split] += 1
        if record.range_m is not None:
            range_count += 1
    summary = {
        "classes": names,
        "images": counts,
        "images_with_labels": labeled,
        "images_with_range": range_count,
    }
    with (out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize LRDDv2 into Ultralytics YOLO layout.")
    parser.add_argument("--raw", type=Path, required=True, help="LRDDv2 directory or authorized archive.")
    parser.add_argument("--out", type=Path, required=True, help="Output YOLO dataset directory.")
    parser.add_argument("--dataset-name", default="lrddv2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--link-mode", choices=["hardlink", "symlink", "copy"], default="hardlink")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_yaml = prepare_lrddv2(
        raw=args.raw,
        out=args.out,
        dataset_name=args.dataset_name,
        seed=args.seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        link_mode=args.link_mode,
    )
    print(data_yaml)


if __name__ == "__main__":
    main()
