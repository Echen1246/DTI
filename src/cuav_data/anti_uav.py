from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import cv2
import yaml
from tqdm import tqdm

CLASS_NAMES = ["friendly_quad", "unknown_uav", "bird", "airplane", "helicopter"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}


def prepare_anti_uav(
    raw: Path,
    out: Path,
    frame_stride: int = 5,
    modality: str = "visible",
    seed: int = 42,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    max_sequences: int | None = None,
) -> Path:
    raw = raw.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    sequences = _discover_sequences(raw, modality=modality)
    if max_sequences is not None:
        sequences = sequences[:max_sequences]
    missing_split_sequences = [seq["sequence_id"] for seq in sequences if seq["split"] == "unknown"]
    split_by_sequence = _assign_splits(missing_split_sequences, seed, val_fraction, test_fraction)
    manifest_rows = []
    summary = {"classes": CLASS_NAMES, "sequences": len(sequences), "images": {"train": 0, "val": 0, "test": 0}}

    for seq in tqdm(sequences, desc="Converting Anti-UAV"):
        split = seq["split"] if seq["split"] != "unknown" else split_by_sequence[seq["sequence_id"]]
        labels = _read_tracking_json(seq["label_path"])
        if seq["video_path"]:
            written = _convert_video_sequence(seq, labels, out, split, frame_stride, manifest_rows)
        else:
            written = _convert_image_sequence(seq, labels, out, split, frame_stride, manifest_rows)
        summary["images"][split] += written

    data_yaml = out / "data.yaml"
    data_yaml.write_text(
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
    with (out / "tracking_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image", "split", "source", "sequence_id", "frame_id", "original_media"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return data_yaml


def _discover_sequences(raw: Path, modality: str = "visible") -> list[dict[str, Any]]:
    sequences = []
    modality = modality.lower()
    for label_path in raw.rglob("*.json"):
        if modality and _is_modality_label(label_path) and label_path.stem.lower() != modality:
            continue
        try:
            payload = json.loads(label_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not _looks_like_tracking_payload(payload):
            continue
        seq_dir = label_path.parent
        sequence_id = "__".join(label_path.relative_to(raw).with_suffix("").parts)
        video_path = _choose_video(seq_dir)
        image_dir = None if video_path else _choose_image_dir(seq_dir)
        if not video_path and not image_dir:
            continue
        sequences.append(
            {
                "sequence_id": sequence_id,
                "split": _infer_split(label_path),
                "label_path": label_path,
                "video_path": video_path,
                "image_dir": image_dir,
            }
        )
    return sorted(sequences, key=lambda x: x["sequence_id"])


def _is_modality_label(label_path: Path) -> bool:
    return label_path.stem.lower() in {"visible", "infrared", "ir", "rgb"}


def _looks_like_tracking_payload(payload: dict[str, Any]) -> bool:
    return any(key in payload for key in ("gt_rect", "bbox", "bboxes", "exist", "exists"))


def _choose_video(seq_dir: Path) -> Path | None:
    videos = [p for p in seq_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES]
    if not videos:
        return None
    preferred = [p for p in videos if any(token in p.name.lower() for token in ("rgb", "visible", "vis"))]
    non_ir = [p for p in videos if "ir" not in p.name.lower() and "infra" not in p.name.lower()]
    return (preferred or non_ir or videos)[0]


def _infer_split(path: Path) -> str:
    for part in path.parts:
        lowered = part.lower()
        if lowered in {"train", "val", "test"}:
            return lowered
        if lowered in {"valid", "validation"}:
            return "val"
    return "unknown"


def _choose_image_dir(seq_dir: Path) -> Path | None:
    candidates = [seq_dir, *[p for p in seq_dir.iterdir() if p.is_dir()]]
    for candidate in candidates:
        if any(p.suffix.lower() in IMAGE_SUFFIXES for p in candidate.iterdir() if p.is_file()):
            return candidate
    return None


def _read_tracking_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    boxes = payload.get("gt_rect") or payload.get("bbox") or payload.get("bboxes") or []
    exists = payload.get("exist") or payload.get("exists")
    if exists is None:
        exists = [1] * len(boxes)
    return {"boxes": boxes, "exists": exists}


def _convert_video_sequence(
    seq: dict[str, Any],
    labels: dict[str, Any],
    out: Path,
    split: str,
    frame_stride: int,
    manifest_rows: list[dict[str, Any]],
) -> int:
    cap = cv2.VideoCapture(str(seq["video_path"]))
    if not cap.isOpened():
        return 0
    written = 0
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % frame_stride != 0:
            frame_idx += 1
            continue
        if not _exists_at(labels, frame_idx):
            frame_idx += 1
            continue
        bbox = _box_at(labels, frame_idx)
        if bbox is None:
            frame_idx += 1
            continue
        height, width = frame.shape[:2]
        yolo = _bbox_to_yolo(bbox, width, height)
        if yolo is None:
            frame_idx += 1
            continue
        stem = f"anti_uav__{seq['sequence_id']}__f{frame_idx + 1:06d}"
        image_rel, label_rel = _write_frame(out, split, stem, frame, yolo)
        manifest_rows.append(_manifest_row(image_rel, split, seq, frame_idx + 1))
        written += 1
        frame_idx += 1
    cap.release()
    return written


def _convert_image_sequence(
    seq: dict[str, Any],
    labels: dict[str, Any],
    out: Path,
    split: str,
    frame_stride: int,
    manifest_rows: list[dict[str, Any]],
) -> int:
    images = sorted(p for p in seq["image_dir"].iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    written = 0
    for frame_idx, image_path in enumerate(images):
        if frame_idx % frame_stride != 0 or not _exists_at(labels, frame_idx):
            continue
        bbox = _box_at(labels, frame_idx)
        if bbox is None:
            continue
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        height, width = frame.shape[:2]
        yolo = _bbox_to_yolo(bbox, width, height)
        if yolo is None:
            continue
        stem = f"anti_uav__{seq['sequence_id']}__f{frame_idx + 1:06d}"
        image_rel, label_rel = _write_frame(out, split, stem, frame, yolo)
        manifest_rows.append(_manifest_row(image_rel, split, seq, frame_idx + 1))
        written += 1
    return written


def _exists_at(labels: dict[str, Any], frame_idx: int) -> bool:
    exists = labels["exists"]
    if frame_idx >= len(exists):
        return True
    value = exists[frame_idx]
    if isinstance(value, list):
        value = value[0] if value else 0
    return bool(value)


def _box_at(labels: dict[str, Any], frame_idx: int) -> tuple[float, float, float, float] | None:
    boxes = labels["boxes"]
    if frame_idx >= len(boxes):
        return None
    box = boxes[frame_idx]
    if not box or len(box) < 4:
        return None
    return float(box[0]), float(box[1]), float(box[2]), float(box[3])


def _write_frame(
    out: Path,
    split: str,
    stem: str,
    frame: Any,
    yolo: tuple[float, float, float, float],
) -> tuple[Path, Path]:
    image_rel = Path("images") / split / f"{stem}.jpg"
    label_rel = Path("labels") / split / f"{stem}.txt"
    cv2.imwrite(str(out / image_rel), frame)
    (out / label_rel).write_text(f"1 {yolo[0]:.8f} {yolo[1]:.8f} {yolo[2]:.8f} {yolo[3]:.8f}\n", encoding="utf-8")
    return image_rel, label_rel


def _manifest_row(image_rel: Path, split: str, seq: dict[str, Any], frame_id: int) -> dict[str, Any]:
    media = seq["video_path"] or seq["image_dir"]
    return {
        "image": str(image_rel),
        "split": split,
        "source": "anti_uav",
        "sequence_id": seq["sequence_id"],
        "frame_id": frame_id,
        "original_media": str(media),
    }


def _assign_splits(sequence_ids: list[str], seed: int, val_fraction: float, test_fraction: float) -> dict[str, str]:
    rng = random.Random(seed)
    sequence_ids = list(sequence_ids)
    rng.shuffle(sequence_ids)
    n = len(sequence_ids)
    n_test = int(n * test_fraction)
    n_val = int(n * val_fraction)
    result = {}
    for sequence_id in sequence_ids[:n_test]:
        result[sequence_id] = "test"
    for sequence_id in sequence_ids[n_test : n_test + n_val]:
        result[sequence_id] = "val"
    for sequence_id in sequence_ids[n_test + n_val :]:
        result[sequence_id] = "train"
    return result


def _bbox_to_yolo(
    bbox: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float] | None:
    x, y, w, h = bbox
    x = max(0.0, min(float(image_width), x))
    y = max(0.0, min(float(image_height), y))
    w = max(0.0, min(float(image_width) - x, w))
    h = max(0.0, min(float(image_height) - y, h))
    if w <= 1 or h <= 1:
        return None
    return ((x + w / 2) / image_width, (y + h / 2) / image_height, w / image_width, h / image_height)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Anti-UAV tracking data to YOLO frames.")
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--modality", default="visible")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--max-sequences", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        prepare_anti_uav(
            raw=args.raw,
            out=args.out,
            frame_stride=args.frame_stride,
            modality=args.modality,
            seed=args.seed,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            max_sequences=args.max_sequences,
        )
    )


if __name__ == "__main__":
    main()
