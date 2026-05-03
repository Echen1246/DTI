from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import yaml
from PIL import Image
from tqdm import tqdm

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CLASS_NAMES = ["friendly_quad", "unknown_uav", "bird", "airplane", "helicopter"]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
ALIASES = {
    "uav": "unknown_uav",
    "drone": "unknown_uav",
    "unknown_uav": "unknown_uav",
    "quadrotor": "unknown_uav",
    "friendly_quad": "friendly_quad",
    "bird": "bird",
    "airplane": "airplane",
    "aeroplane": "airplane",
    "plane": "airplane",
    "helicopter": "helicopter",
}


def prepare_detection_dataset(
    raw: Path,
    out: Path,
    source_name: str,
    default_class: str = "unknown_uav",
    format_hint: str | None = None,
    seed: int = 42,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    link_mode: str = "hardlink",
) -> Path:
    raw = raw.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    format_hint = format_hint.lower() if format_hint else None
    if format_hint == "yolo" or (format_hint is None and _looks_like_yolo(raw)):
        print(f"Reading {source_name} as YOLO annotations", flush=True)
        records = _read_yolo_records(raw, default_class)
    elif format_hint == "coco":
        print(f"Reading {source_name} as COCO annotations", flush=True)
        records = _read_coco_records(_find_coco_jsons(raw), raw, default_class)
    elif format_hint == "voc":
        print(f"Reading {source_name} as VOC annotations", flush=True)
        records = _read_voc_records(raw, default_class)
    else:
        coco_files = _find_coco_jsons(raw)
        if coco_files:
            print(f"Reading {source_name} as COCO annotations", flush=True)
            records = _read_coco_records(coco_files, raw, default_class)
        else:
            print(f"Reading {source_name} as VOC annotations", flush=True)
            records = _read_voc_records(raw, default_class)

    if not records:
        raise ValueError(f"No supported detection annotations found under {raw}")

    records = _fill_missing_splits(records, seed, val_fraction, test_fraction)
    summary = {"source": source_name, "images": {"train": 0, "val": 0, "test": 0}, "labels": 0}
    for record in tqdm(records, desc=f"Converting {source_name}"):
        split = record["split"]
        image = record["image"]
        dest_image = out / "images" / split / f"{source_name}__{_safe_stem(image, raw)}{image.suffix.lower()}"
        dest_label = out / "labels" / split / f"{dest_image.stem}.txt"
        _place_file(image, dest_image, link_mode)
        dest_label.write_text("".join(record["labels"]), encoding="utf-8")
        summary["images"][split] += 1
        summary["labels"] += len(record["labels"])

    (out / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(out.resolve()),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": {i: name for i, name in enumerate(CLASS_NAMES)},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out / "data.yaml"


def _looks_like_yolo(root: Path) -> bool:
    return any(p.name == "labels" for p in root.rglob("labels"))


def _read_yolo_records(root: Path, default_class: str) -> list[dict[str, Any]]:
    records = []
    data_yaml = next(iter(root.rglob("data.yaml")), None)
    names = _load_names(data_yaml) if data_yaml else ["drone"]
    for image in _iter_images(root):
        if "/labels/" in str(image):
            continue
        split = _infer_split(image)
        label = _label_for_yolo_image(image)
        labels = []
        if label.exists():
            for line in label.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                src_name = names[int(float(parts[0]))] if int(float(parts[0])) < len(names) else default_class
                target = ALIASES.get(src_name.lower(), ALIASES.get(default_class.lower(), "drone"))
                labels.append(" ".join([str(CLASS_TO_ID[target]), *parts[1:5]]) + "\n")
        records.append({"image": image, "split": split, "labels": labels})
    return records


def _find_coco_jsons(root: Path) -> list[Path]:
    found = []
    for path in root.rglob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict) and "images" in data and "annotations" in data:
            found.append(path)
    return found


def _read_coco_records(paths: list[Path], root: Path, default_class: str) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        categories = {cat["id"]: ALIASES.get(str(cat.get("name", default_class)).lower(), default_class) for cat in data.get("categories", [])}
        anns_by_image: dict[int, list[dict[str, Any]]] = {}
        for ann in data.get("annotations", []):
            anns_by_image.setdefault(int(ann["image_id"]), []).append(ann)
        for image_info in data.get("images", []):
            image = _resolve_image(root, path.parent, str(image_info["file_name"]))
            width, height = _image_size(image, image_info)
            labels = []
            for ann in anns_by_image.get(int(image_info["id"]), []):
                bbox = ann.get("bbox")
                if not bbox or len(bbox) < 4:
                    continue
                target = categories.get(ann.get("category_id"), default_class)
                if target not in CLASS_TO_ID:
                    continue
                x, y, w, h = map(float, bbox[:4])
                labels.append(_yolo_line(CLASS_TO_ID[target], x, y, w, h, width, height))
            records.append({"image": image, "split": _infer_split(image), "labels": labels})
    return records


def _read_voc_records(root: Path, default_class: str) -> list[dict[str, Any]]:
    records = []
    image_index = _build_image_index(root)
    for xml_path in root.rglob("*.xml"):
        try:
            xml = ET.parse(xml_path).getroot()
        except Exception:
            continue
        filename = xml.findtext("filename")
        if not filename:
            continue
        image = _resolve_image(root, xml_path.parent, filename, image_index)
        width, height = _voc_size(xml) or _image_size(image, {})
        labels = []
        for obj in xml.findall("object"):
            name = obj.findtext("name") or default_class
            target = ALIASES.get(name.lower(), ALIASES.get(default_class.lower(), "drone"))
            box = obj.find("bndbox")
            if target not in CLASS_TO_ID or box is None:
                continue
            xmin = float(box.findtext("xmin", "0"))
            ymin = float(box.findtext("ymin", "0"))
            xmax = float(box.findtext("xmax", "0"))
            ymax = float(box.findtext("ymax", "0"))
            labels.append(_yolo_line(CLASS_TO_ID[target], xmin, ymin, xmax - xmin, ymax - ymin, width, height))
        records.append({"image": image, "split": _infer_split(image), "labels": labels})
    return records


def _voc_size(xml: ET.Element) -> tuple[float, float] | None:
    size = xml.find("size")
    if size is None:
        return None
    width = size.findtext("width")
    height = size.findtext("height")
    if not width or not height:
        return None
    try:
        return float(width), float(height)
    except ValueError:
        return None


def _build_image_index(root: Path) -> dict[str, Path]:
    index = {}
    for image in _iter_images(root):
        index.setdefault(image.name, image)
    return index


def _fill_missing_splits(records: list[dict[str, Any]], seed: int, val_fraction: float, test_fraction: float) -> list[dict[str, Any]]:
    missing = [record for record in records if record["split"] == "unknown"]
    rng = random.Random(seed)
    rng.shuffle(missing)
    n_test = int(len(missing) * test_fraction)
    n_val = int(len(missing) * val_fraction)
    for record in missing[:n_test]:
        record["split"] = "test"
    for record in missing[n_test : n_test + n_val]:
        record["split"] = "val"
    for record in missing[n_test + n_val :]:
        record["split"] = "train"
    for record in records:
        if record["split"] == "valid":
            record["split"] = "val"
    return records


def _load_names(data_yaml: Path) -> list[str]:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = data.get("names", ["drone"])
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names, key=lambda x: int(x) if str(x).isdigit() else str(x))]
    return [str(name) for name in names]


def _iter_images(root: Path):
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def _infer_split(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    if "train" in parts:
        return "train"
    if "val" in parts or "valid" in parts or "validation" in parts:
        return "val"
    if "test" in parts:
        return "test"
    return "unknown"


def _label_for_yolo_image(image: Path) -> Path:
    parts = list(image.parts)
    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image.with_suffix(".txt")


def _resolve_image(
    root: Path,
    annotation_dir: Path,
    filename: str,
    image_index: dict[str, Path] | None = None,
) -> Path:
    for candidate in (
        annotation_dir / filename,
        annotation_dir.parent / "img" / filename,
        annotation_dir.parent / "images" / filename,
        root / filename,
        root / "images" / filename,
    ):
        if candidate.exists():
            return candidate
    if image_index:
        indexed = image_index.get(Path(filename).name)
        if indexed:
            return indexed
    matches = list(root.rglob(Path(filename).name))
    if matches:
        return matches[0]
    raise FileNotFoundError(filename)


def _image_size(image: Path, image_info: dict[str, Any]) -> tuple[float, float]:
    if image_info.get("width") and image_info.get("height"):
        return float(image_info["width"]), float(image_info["height"])
    with Image.open(image) as img:
        return float(img.width), float(img.height)


def _yolo_line(class_id: int, x: float, y: float, w: float, h: float, image_width: float, image_height: float) -> str:
    return (
        f"{class_id} "
        f"{((x + w / 2) / image_width):.8f} {((y + h / 2) / image_height):.8f} "
        f"{(w / image_width):.8f} {(h / image_height):.8f}\n"
    )


def _safe_stem(path: Path, root: Path) -> str:
    try:
        return "__".join(path.relative_to(root).with_suffix("").parts)
    except ValueError:
        return path.stem


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a detection dataset to canonical YOLO layout.")
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--default-class", default="unknown_uav")
    parser.add_argument("--format-hint", choices=["yolo", "coco", "voc"])
    parser.add_argument("--link-mode", choices=["hardlink", "symlink", "copy"], default="hardlink")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        prepare_detection_dataset(
            raw=args.raw,
            out=args.out,
            source_name=args.source_name,
            default_class=args.default_class,
            format_hint=args.format_hint,
            link_mode=args.link_mode,
        )
    )


if __name__ == "__main__":
    main()
