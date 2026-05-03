from __future__ import annotations

import subprocess
from pathlib import Path

import modal

APP_NAME = "cuas-yolo11-training"
DATA_VOL_NAME = "cuas-data"
RUNS_VOL_NAME = "cuas-runs"
WORKDIR = "/workspace"
YOLO11X_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11x.pt"

app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(DATA_VOL_NAME, create_if_missing=True)
runs_volume = modal.Volume.from_name(RUNS_VOL_NAME, create_if_missing=True)

data_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .uv_pip_install(
        "gdown>=5.2.0",
        "opencv-python-headless>=4.10.0",
        "pillow>=10.4.0",
        "pyyaml>=6.0.2",
        "requests>=2.32.0",
        "tqdm>=4.66.0",
    )
    .env({"PYTHONPATH": f"{WORKDIR}/src"})
    .add_local_dir("src", f"{WORKDIR}/src", copy=True)
)

roboflow_secret = modal.Secret.from_name("roboflow-api")

train_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .uv_pip_install(
        "torch==2.7.1",
        "torchvision==0.22.1",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .uv_pip_install(
        "ultralytics>=8.3.0",
        "sahi>=0.11.20",
        "opencv-python-headless>=4.10.0",
        "pandas>=2.2.0",
        "pyyaml>=6.0.2",
        "tqdm>=4.66.0",
        "pillow>=10.4.0",
        "lap>=0.5.12",
        "scipy>=1.13.0",
    )
    .env(
        {
            "PYTHONPATH": f"{WORKDIR}/src",
            "WANDB_MODE": "disabled",
            "ULTRALYTICS_SETTINGS": "/tmp/ultralytics-settings.json",
        }
    )
    .add_local_dir("src", f"{WORKDIR}/src", copy=True)
    .add_local_dir("configs", f"{WORKDIR}/configs", copy=True)
)


@app.function(
    image=data_image,
    volumes={"/data": data_volume},
    timeout=24 * 60 * 60,
)
def ingest_lrddv2(
    remote_archive: str = "raw/lrddv2/LRDDv2.zip",
    output_dir: str = "datasets/lrddv2",
    link_mode: str = "hardlink",
) -> str:
    from cuav_data.lrddv2 import prepare_lrddv2

    data_volume.reload()
    archive_path = Path("/data") / remote_archive
    out_path = Path("/data") / output_dir
    data_yaml = prepare_lrddv2(
        raw=archive_path,
        out=out_path,
        dataset_name="lrddv2",
        link_mode=link_mode,
    )
    data_volume.commit()
    return str(data_yaml)


@app.function(
    image=train_image,
    gpu="B200:8",
    volumes={"/data": data_volume, "/runs": runs_volume},
    timeout=24 * 60 * 60,
)
def train_yolo11x_p2(
    data_yaml: str = "/data/datasets/open-cuas/data.yaml",
    run_name: str = "open-cuas-yolo11x-p2",
    epochs: int = 200,
    imgsz: int = 1536,
    batch: str = "64",
    workers: int = 16,
    patience: int = 50,
    resume: bool = False,
) -> str:
    data_volume.reload()
    runs_volume.reload()

    model_yaml = f"{WORKDIR}/configs/models/yolo11x-p2.yaml"
    pretrained_ckpt = _ensure_yolo11x_checkpoint()
    project_dir = "/runs/yolo"
    device = "0,1,2,3,4,5,6,7"

    if resume:
        resume_ckpt = Path(project_dir) / run_name / "weights" / "last.pt"
        cmd = ["yolo", "detect", "train", "resume", f"model={resume_ckpt}"]
    else:
        cmd = [
            "yolo",
            "detect",
            "train",
            f"model={model_yaml}",
            f"pretrained={pretrained_ckpt}",
            f"data={data_yaml}",
            f"epochs={epochs}",
            f"imgsz={imgsz}",
            f"batch={batch}",
            f"workers={workers}",
            f"patience={patience}",
            f"project={project_dir}",
            f"name={run_name}",
            "exist_ok=True",
            f"device={device}",
            "optimizer=auto",
            "amp=True",
            "cache=disk",
            "cos_lr=True",
            "close_mosaic=20",
            "multi_scale=0.25",
            "save_period=10",
            "plots=True",
        ]

    subprocess.run(cmd, cwd=WORKDIR, check=True)
    runs_volume.commit()
    data_volume.commit()
    return str(Path(project_dir) / run_name)


def _ensure_yolo11x_checkpoint() -> str:
    import urllib.request

    dest = Path("/tmp/ultralytics_weights/yolo11x.pt")
    min_bytes = 100 * 1024 * 1024
    if dest.exists() and dest.stat().st_size >= min_bytes:
        print(f"Using cached pretrained checkpoint: {dest} ({dest.stat().st_size} bytes)", flush=True)
        return str(dest)

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".pt.tmp")
    if tmp.exists():
        tmp.unlink()

    print(f"Downloading pretrained checkpoint once before DDP: {YOLO11X_URL}", flush=True)
    with urllib.request.urlopen(YOLO11X_URL, timeout=120) as response:
        with tmp.open("wb") as f:
            while True:
                chunk = response.read(16 * 1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                print(f"Downloaded {tmp.stat().st_size} bytes", flush=True)

    size = tmp.stat().st_size
    if size < min_bytes:
        raise RuntimeError(f"Downloaded checkpoint is too small: {size} bytes")
    tmp.replace(dest)
    print(f"Pretrained checkpoint ready: {dest} ({size} bytes)", flush=True)
    return str(dest)


@app.function(
    image=data_image,
    volumes={"/data": data_volume, "/runs": runs_volume},
    timeout=30 * 60,
)
def status() -> dict[str, list[str]]:
    data_volume.reload()
    runs_volume.reload()
    datasets = sorted(str(p.relative_to("/data")) for p in Path("/data/datasets").glob("*") if p.exists())
    runs = sorted(str(p.relative_to("/runs")) for p in Path("/runs/yolo").glob("*") if p.exists())
    return {"datasets": datasets, "runs": runs}


@app.function(
    image=train_image,
    gpu="B200",
    volumes={"/data": data_volume, "/runs": runs_volume},
    timeout=45 * 60,
)
def build_validation_demo(
    sequence_prefix: str = "anti_uav300__anti_uav__val__20190925_101846_1_4__visible",
    run_name: str = "open-cuas-yolo11x-p2",
    output_name: str = "real_validation_multi_target.mp4",
    max_frames: int = 240,
    extra_targets: int = 4,
    conf: float = 0.18,
    iou: float = 0.55,
    imgsz: int = 1536,
    draw_scenario: bool = False,
    fps: float = 10.0,
) -> str:
    import cv2
    import numpy as np
    from ultralytics import YOLO

    data_volume.reload()
    runs_volume.reload()

    images_dir = Path("/data/datasets/open-cuas/images/val")
    labels_dir = Path("/data/datasets/open-cuas/labels/val")
    image_paths = sorted(images_dir.glob(f"{sequence_prefix}*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No validation frames matched prefix {sequence_prefix!r}")
    image_paths = image_paths[:max_frames]

    weights = Path("/runs/yolo") / run_name / "weights" / "best.pt"
    if not weights.exists():
        raise FileNotFoundError(weights)
    model = YOLO(str(weights))

    first = cv2.imread(str(image_paths[0]))
    if first is None:
        raise RuntimeError(f"Could not read first frame: {image_paths[0]}")
    height, width = first.shape[:2]

    out_dir = Path("/runs/demos")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / output_name
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video writer: {out_path}")

    for frame_idx, image_path in enumerate(image_paths):
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        scenario_boxes = _read_yolo_boxes(labels_dir / f"{image_path.stem}.txt", width, height)
        drone_patch = _extract_patch(frame, scenario_boxes)
        scenario_boxes.extend(
            _add_extra_targets(
                frame=frame,
                frame_idx=frame_idx,
                count=extra_targets,
                patch=drone_patch,
            )
        )

        result = model.predict(frame, conf=conf, iou=iou, imgsz=imgsz, max_det=50, verbose=False)[0]
        model_boxes = []
        for box in result.boxes:
            class_id = int(box.cls[0])
            name = str(result.names.get(class_id, f"class_{class_id}"))
            if name not in {"unknown_uav", "friendly_quad", "drone"} and class_id not in {0, 1}:
                continue
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            model_boxes.append((x1, y1, x2, y2, float(box.conf[0]), name))

        _draw_demo_frame(
            frame=frame,
            frame_idx=frame_idx,
            scenario_boxes=scenario_boxes,
            model_boxes=model_boxes,
            conf=conf,
            draw_scenario=draw_scenario,
        )
        writer.write(frame)

    writer.release()
    runs_volume.commit()
    return str(out_path)


@app.function(
    image=train_image,
    gpu="B200",
    volumes={"/data": data_volume, "/runs": runs_volume},
    timeout=45 * 60,
)
def annotate_video_demo(
    remote_source: str,
    run_name: str = "open-cuas-yolo11x-p2",
    output_name: str = "m2-res_720p_annotated",
    conf: float = 0.08,
    imgsz: int = 1536,
    iou: float = 0.55,
    side_panel: bool = False,
) -> str:
    from dti_demo.platform import CameraPose, run_demo

    data_volume.reload()
    runs_volume.reload()

    source = Path("/data") / remote_source
    if not source.exists():
        raise FileNotFoundError(source)
    weights = Path("/runs/yolo") / run_name / "weights" / "best.pt"
    if not weights.exists():
        raise FileNotFoundError(weights)

    output_dir = Path("/runs/demos") / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    run_demo(
        output_dir=output_dir,
        weights=weights,
        source=source,
        conf=conf,
        imgsz=imgsz,
        iou=iou,
        side_panel=side_panel,
        camera_pose=CameraPose(),
    )
    runs_volume.commit()
    return str(output_dir)


def _read_yolo_boxes(label_path: Path, width: int, height: int) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        _, xc, yc, bw, bh = parts[:5]
        xc_f = float(xc) * width
        yc_f = float(yc) * height
        bw_f = float(bw) * width
        bh_f = float(bh) * height
        x1 = int(max(0, xc_f - bw_f / 2))
        y1 = int(max(0, yc_f - bh_f / 2))
        x2 = int(min(width - 1, xc_f + bw_f / 2))
        y2 = int(min(height - 1, yc_f + bh_f / 2))
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))
    return boxes


def _extract_patch(frame, boxes: list[tuple[int, int, int, int]]):
    import cv2
    import numpy as np

    if boxes:
        x1, y1, x2, y2 = max(boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
        pad = 4
        y1 = max(0, y1 - pad)
        x1 = max(0, x1 - pad)
        y2 = min(frame.shape[0], y2 + pad)
        x2 = min(frame.shape[1], x2 + pad)
        patch = frame[y1:y2, x1:x2].copy()
        if patch.size:
            return patch
    patch = np.zeros((18, 34, 3), dtype=np.uint8)
    cv2.line(patch, (2, 9), (32, 9), (35, 35, 35), 2)
    cv2.line(patch, (17, 2), (17, 16), (35, 35, 35), 2)
    cv2.circle(patch, (17, 9), 3, (20, 20, 20), -1)
    return patch


def _add_extra_targets(frame, frame_idx: int, count: int, patch) -> list[tuple[int, int, int, int]]:
    import math

    import cv2
    import numpy as np

    height, width = frame.shape[:2]
    boxes: list[tuple[int, int, int, int]] = []
    if count <= 0:
        return boxes

    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
    if mask.mean() < 2:
        mask = np.full(gray.shape, 210, dtype=np.uint8)

    for idx in range(count):
        scale = 0.55 + 0.12 * ((idx + frame_idx // 20) % 4)
        target_w = max(10, int(patch.shape[1] * scale))
        target_h = max(8, int(patch.shape[0] * scale))
        sprite = cv2.resize(patch, (target_w, target_h), interpolation=cv2.INTER_AREA)
        sprite_mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_AREA)

        x = int(width * (0.18 + 0.13 * idx) + frame_idx * (1.1 + 0.25 * idx)) % max(1, width - target_w - 2)
        y = int(height * (0.18 + 0.08 * (idx % 3)) + 18 * math.sin(frame_idx / 24 + idx))
        y = max(5, min(height - target_h - 5, y))

        roi = frame[y : y + target_h, x : x + target_w]
        alpha = (sprite_mask.astype(float) / 255.0)[:, :, None] * 0.72
        roi[:] = (roi.astype(float) * (1.0 - alpha) + sprite.astype(float) * alpha).astype(np.uint8)
        boxes.append((x, y, x + target_w, y + target_h))
    return boxes


def _draw_demo_frame(
    frame,
    frame_idx: int,
    scenario_boxes: list[tuple[int, int, int, int]],
    model_boxes: list[tuple[int, int, int, int, float, str]],
    conf: float,
    draw_scenario: bool = False,
) -> None:
    import cv2

    if draw_scenario:
        for idx, (x1, y1, x2, y2) in enumerate(scenario_boxes, start=1):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (70, 190, 255), 1)
            cv2.putText(
                frame,
                f"scenario T{idx:02d}",
                (x1, max(14, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (70, 190, 255),
                1,
                cv2.LINE_AA,
            )

    for idx, (x1, y1, x2, y2, score, name) in enumerate(model_boxes, start=1):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (70, 230, 140), 2)
        cv2.putText(
            frame,
            f"model {idx:02d} {name} {score:.2f}",
            (x1, max(18, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (70, 230, 140),
            2,
            cv2.LINE_AA,
        )

    rough_ranges = [_rough_range_from_box(x1, x2) for x1, _, x2, _, _, _ in model_boxes]
    nearest = min(rough_ranges) if rough_ranges else None
    average = sum(rough_ranges) / len(rough_ranges) if rough_ranges else None
    cv2.rectangle(frame, (18, 18), (500, 126), (10, 14, 18), -1)
    cv2.putText(frame, "DTI CAMERA FEED - MODEL DETECTIONS", (30, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (235, 240, 245), 2, cv2.LINE_AA)
    cv2.putText(frame, f"targets: {len(model_boxes)} | threshold: {conf:.2f} | frame: {frame_idx:04d}", (30, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (170, 185, 195), 1, cv2.LINE_AA)
    range_text = "range: n/a" if average is None else f"avg range: {average:.0f}m | nearest: {nearest:.0f}m"
    cv2.putText(frame, range_text, (30, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (130, 220, 180), 1, cv2.LINE_AA)


def _rough_range_from_box(x1: int, x2: int, focal_px: float = 1800.0, width_m: float = 0.6) -> float:
    bbox_width = max(1, x2 - x1)
    return focal_px * width_m / bbox_width


@app.function(
    image=data_image,
    volumes={"/data": data_volume},
    timeout=24 * 60 * 60,
)
def download_gdrive(file_id: str, output_path: str) -> str:
    import gdown

    data_volume.reload()
    dest = Path("/data") / output_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Google Drive file {file_id} to {dest}", flush=True)
    gdown.download(id=file_id, output=str(dest), quiet=False)
    print(f"Downloaded {dest} ({dest.stat().st_size} bytes). Committing volume...", flush=True)
    data_volume.commit()
    print("Volume committed.", flush=True)
    return str(dest)


@app.function(
    image=data_image,
    volumes={"/data": data_volume},
    timeout=24 * 60 * 60,
)
def download_url(url: str, output_path: str) -> str:
    import requests

    data_volume.reload()
    dest = Path("/data") / output_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading URL to {dest}", flush=True)
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with dest.open("wb") as f:
            for chunk in response.iter_content(chunk_size=16 * 1024 * 1024):
                if chunk:
                    f.write(chunk)
                    print(f"Wrote {dest.stat().st_size} bytes", flush=True)
    print(f"Downloaded {dest} ({dest.stat().st_size} bytes). Committing volume...", flush=True)
    data_volume.commit()
    print("Volume committed.", flush=True)
    return str(dest)


@app.function(
    image=data_image,
    secrets=[roboflow_secret],
    volumes={"/data": data_volume},
    timeout=24 * 60 * 60,
)
def download_roboflow(
    workspace: str = "drone-a7lpy",
    project: str = "drones-yolo11-a",
    version: int = 16,
    export_format: str = "yolov11",
    output_path: str = "raw/roboflow/drones-yolo11-a.zip",
) -> str:
    import os
    import zipfile

    import requests

    api_keys = [
        key
        for key in (
            os.environ.get("ROBOFLOW_API_KEY"),
            os.environ.get("ROBOFLOW_PUBLISHABLE_API_KEY"),
        )
        if key
    ]
    if not api_keys:
        raise ValueError("ROBOFLOW_API_KEY is not configured in Modal secret roboflow-api")

    data_volume.reload()
    dest = Path("/data") / output_path
    dest.parent.mkdir(parents=True, exist_ok=True)

    api_export_formats = [export_format, "yolov8", "yolov5pytorch"]
    candidates = []
    for key in api_keys:
        candidates.extend(
            [
                {
                    "kind": "download",
                    "url": f"https://universe.roboflow.com/{workspace}/{project}/dataset/{version}/download/{fmt}",
                    "params": {"api_key": key},
                }
                for fmt in api_export_formats
            ]
        )
        candidates.extend(
            [
                {
                    "kind": "api",
                    "url": f"https://api.roboflow.com/{workspace}/{project}/{version}/{fmt}",
                    "params": {"api_key": key},
                }
                for fmt in api_export_formats
            ]
        )
    last_error = None
    for candidate in candidates:
        print(
            f"Requesting Roboflow {candidate['kind']} export for {workspace}/{project}:{version}",
            flush=True,
        )
        try:
            response = requests.get(candidate["url"], params=candidate["params"], stream=True, timeout=120)
            if not response.ok:
                body = response.text[:400].replace("\n", " ")
                raise RuntimeError(f"HTTP {response.status_code}: {body}")
            content_type = response.headers.get("content-type", "")
            download_url = None
            if "application/json" in content_type:
                payload = response.json()
                download_url = (
                    payload.get("export", {}).get("link")
                    or payload.get("export", {}).get("url")
                    or payload.get("link")
                    or payload.get("url")
                )
                if not download_url:
                    raise RuntimeError(f"Roboflow JSON response did not include an export link: {payload.keys()}")
                response.close()
                response = requests.get(download_url, stream=True, timeout=120)
                if not response.ok:
                    body = response.text[:400].replace("\n", " ")
                    raise RuntimeError(f"download HTTP {response.status_code}: {body}")
            with response:
                with dest.open("wb") as f:
                    for chunk in response.iter_content(chunk_size=16 * 1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            print(f"Wrote {dest.stat().st_size} bytes", flush=True)
            with zipfile.ZipFile(dest) as zf:
                names = zf.namelist()
                if not any(name.endswith("data.yaml") for name in names):
                    raise ValueError(f"Downloaded Roboflow file has no data.yaml: {dest}")
            data_volume.commit()
            return str(dest)
        except Exception as exc:
            last_error = exc
            if dest.exists():
                dest.unlink()
            print(f"Roboflow export attempt failed: {type(exc).__name__}: {exc}", flush=True)
    raise RuntimeError(f"Unable to download Roboflow dataset {workspace}/{project}:{version}") from last_error


@app.function(
    image=data_image,
    volumes={"/data": data_volume},
    timeout=30 * 60,
)
def read_text(remote_path: str, max_chars: int = 12000) -> str:
    data_volume.reload()
    path = Path("/data") / remote_path
    if not path.exists():
        return f"{path} does not exist."
    return path.read_text(encoding="utf-8", errors="replace")[:max_chars]


@app.function(
    image=data_image,
    volumes={"/data": data_volume},
    timeout=30 * 60,
)
def inspect_zip(remote_zip: str, max_entries: int = 120) -> list[str]:
    import zipfile
    from collections import Counter

    data_volume.reload()
    path = Path("/data") / remote_zip
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
    suffix_counts = Counter(Path(name).suffix.lower() or "<dir>" for name in names)
    interesting = [
        name
        for name in names
        if any(token in name.lower() for token in ("ann", "label", "xml", "json", "txt", "gt", "bbox"))
    ]
    return [
        f"total_entries={len(names)}",
        f"suffix_counts={dict(suffix_counts)}",
        "interesting:",
        *interesting[:max_entries],
        "first_entries:",
        *names[:max_entries],
    ]


@app.function(
    image=data_image,
    volumes={"/data": data_volume},
    timeout=30 * 60,
)
def read_zip_text(remote_zip: str, member_suffix: str, max_chars: int = 8000) -> str:
    import zipfile

    data_volume.reload()
    path = Path("/data") / remote_zip
    with zipfile.ZipFile(path) as zf:
        matches = [name for name in zf.namelist() if name.endswith(member_suffix)]
        if not matches:
            return f"No member ending with {member_suffix!r} found."
        name = sorted(matches, key=len)[0]
        text = zf.read(name).decode("utf-8", errors="replace")
    return f"--- {name} ---\n{text[:max_chars]}"


@app.function(
    image=data_image,
    volumes={"/data": data_volume},
    timeout=24 * 60 * 60,
)
def extract_zip(remote_zip: str, output_dir: str) -> str:
    import zipfile

    data_volume.reload()
    src = Path("/data") / remote_zip
    dest = Path("/data") / output_dir
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {src} to {dest}", flush=True)
    with zipfile.ZipFile(src) as zf:
        zf.extractall(dest)
    data_volume.commit()
    return str(dest)


@app.function(
    image=data_image,
    volumes={"/data": data_volume},
    timeout=24 * 60 * 60,
)
def convert_detection(remote_raw_dir: str, output_dir: str, source_name: str) -> str:
    from cuav_data.detection_convert import prepare_detection_dataset
    from cuav_data.validate_yolo import validate_yolo_dataset

    data_volume.reload()
    data_yaml = prepare_detection_dataset(
        raw=Path("/data") / remote_raw_dir,
        out=Path("/data") / output_dir,
        source_name=source_name,
        default_class="unknown_uav",
        link_mode="copy",
    )
    summary = validate_yolo_dataset(data_yaml)
    (Path("/data") / output_dir / "validation_summary.json").write_text(
        __import__("json").dumps(summary, indent=2),
        encoding="utf-8",
    )
    data_volume.commit()
    return str(data_yaml)


@app.function(
    image=data_image,
    volumes={"/data": data_volume},
    timeout=24 * 60 * 60,
)
def prepare_open_cuas(
    output_dir: str = "datasets/open-cuas",
    frame_stride: int = 10,
    max_sequences: int | None = None,
    reset_outputs: bool = True,
) -> dict[str, object]:
    import json
    import shutil
    import zipfile

    from cuav_data.anti_uav import prepare_anti_uav
    from cuav_data.detection_convert import prepare_detection_dataset
    from cuav_data.validate_yolo import validate_yolo_dataset
    from cuav_data.yolo_merge import SourceSpec, merge_yolo_datasets

    class_names = ["friendly_quad", "unknown_uav", "bird", "airplane", "helicopter"]
    data_volume.reload()
    work_root = Path("/tmp/open_cuas_prepare")
    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True, exist_ok=True)

    def extract_once(remote_zip: str, dest_dir: str) -> Path:
        src = Path("/data") / remote_zip
        dest = work_root / dest_dir
        dest.mkdir(parents=True, exist_ok=True)
        print(f"Extracting {src} -> {dest}", flush=True)
        with zipfile.ZipFile(src) as zf:
            zf.extractall(dest)
        return dest

    def clear_generated(path: Path) -> None:
        if reset_outputs and path.exists():
            print(f"Clearing generated dataset directory: {path}", flush=True)
            shutil.rmtree(path)
            print(f"Cleared {path}", flush=True)

    source_specs: list[SourceSpec] = []
    source_yamls: dict[str, str] = {}
    generated_root = Path("/data/datasets") / f"sources-{Path(output_dir).name}-prepared"

    dut_raw = work_root / "raw/dut-anti-uav"
    for split_zip in ("train.zip", "val.zip", "test.zip"):
        extract_once(f"raw/dut-anti-uav/{split_zip}", "raw/dut-anti-uav")
    dut_out = generated_root / "dut-anti-uav"
    clear_generated(dut_out)
    dut_yaml = prepare_detection_dataset(
        raw=dut_raw,
        out=dut_out,
        source_name="dut_anti_uav",
        default_class="unknown_uav",
        format_hint="voc",
        link_mode="hardlink",
    )
    source_yamls["dut_anti_uav"] = str(dut_yaml)
    source_specs.append(
        SourceSpec(
            name="dut_anti_uav",
            root=dut_out,
            class_map={"unknown_uav": "unknown_uav"},
            weight=1.5,
        )
    )

    anti_raw = extract_once("raw/anti-uav300/anti-uav300.zip", "raw/anti-uav300")
    anti_out = generated_root / "anti-uav300"
    clear_generated(anti_out)
    anti_yaml = prepare_anti_uav(
        raw=anti_raw,
        out=anti_out,
        frame_stride=frame_stride,
        modality="visible",
        max_sequences=max_sequences,
    )
    source_yamls["anti_uav300"] = str(anti_yaml)
    source_specs.append(
        SourceSpec(
            name="anti_uav300",
            root=anti_out,
            class_map={"unknown_uav": "unknown_uav"},
            weight=1.25,
        )
    )

    dvb_raw = extract_once("raw/drone-vs-bird/drone-vs-bird.zip", "raw/drone-vs-bird")
    dvb_root = _find_yolo_dataset_root(dvb_raw)
    if dvb_root:
        source_yamls["drone_vs_bird"] = str(dvb_root / "data.yaml")
        source_specs.append(
            SourceSpec(
                name="drone_vs_bird",
                root=dvb_root,
                class_map={
                    "drone": "unknown_uav",
                    "Drone": "unknown_uav",
                    "uav": "unknown_uav",
                    "UAV": "unknown_uav",
                    "bird": "bird",
                    "Bird": "bird",
                },
                weight=1.0,
            )
        )
    else:
        print("Drone-vs-Bird archive extracted, but no YOLO data.yaml root was found.", flush=True)

    roboflow_zip = Path("/data/raw/roboflow/drones-yolo11-a.zip")
    roboflow_candidates = [
        work_root / "raw/roboflow/drones-yolo11-a",
        Path("/data/raw/roboflow/drones-yolo11-a"),
        Path("/data/raw/drones-yolo11-a"),
        Path("/data/raw/roboflow"),
    ]
    if roboflow_zip.exists():
        extract_once("raw/roboflow/drones-yolo11-a.zip", "raw/roboflow/drones-yolo11-a")
    roboflow_root = _first_existing_yolo_root(roboflow_candidates)
    if roboflow_root:
        source_yamls["roboflow_drones_yolo11_a"] = str(roboflow_root / "data.yaml")
        source_specs.append(
            SourceSpec(
                name="roboflow_drones_yolo11_a",
                root=roboflow_root,
                class_map={
                    "0": "unknown_uav",
                    "drone": "unknown_uav",
                    "Drone": "unknown_uav",
                    "uav": "unknown_uav",
                    "UAV": "unknown_uav",
                    "unknown_uav": "unknown_uav",
                    "bird": "bird",
                    "Bird": "bird",
                    "airplane": "airplane",
                    "Airplane": "airplane",
                    "plane": "airplane",
                },
                weight=1.0,
            )
        )
    else:
        print("Roboflow drones-yolo11-a is not present in the volume yet; skipping it.", flush=True)

    merged_out = Path("/data") / output_dir
    clear_generated(merged_out)
    data_yaml = merge_yolo_datasets(
        sources=source_specs,
        out=merged_out,
        class_names=class_names,
        link_mode="hardlink",
    )
    validation = validate_yolo_dataset(data_yaml)
    (merged_out / "validation_summary.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    data_volume.commit()

    return {
        "data_yaml": str(data_yaml),
        "sources": source_yamls,
        "validation": validation,
    }


def _find_yolo_dataset_root(root: Path) -> Path | None:
    candidates = sorted(root.rglob("data.yaml"), key=lambda p: len(p.parts))
    for data_yaml in candidates:
        parent = data_yaml.parent
        if any((parent / split / "images").exists() for split in ("train", "valid", "val", "test")):
            return parent
        if (parent / "images").exists():
            return parent
    return candidates[0].parent if candidates else None


def _first_existing_yolo_root(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if not candidate.exists():
            continue
        if (candidate / "data.yaml").exists():
            return candidate
        found = _find_yolo_dataset_root(candidate)
        if found:
            return found
    return None


@app.local_entrypoint()
def main(
    action: str = "status",
    local_path: str | None = None,
    remote_archive: str = "raw/lrddv2/LRDDv2.zip",
    remote_path: str | None = None,
    file_id: str | None = None,
    url: str | None = None,
    source_name: str = "dataset",
    data_yaml: str = "/data/datasets/open-cuas/data.yaml",
    run_name: str = "open-cuas-yolo11x-p2",
    epochs: int = 200,
    imgsz: int = 1536,
    batch: str = "64",
    resume: bool = False,
    frame_stride: int = 10,
    max_sequences: int | None = None,
    sequence_prefix: str = "anti_uav300__anti_uav__val__20190925_101846_1_4__visible",
    output_name: str = "real_validation_multi_target.mp4",
    max_frames: int = 240,
    extra_targets: int = 4,
    conf: float = 0.18,
    iou: float = 0.55,
    draw_scenario: bool = False,
    fps: float = 10.0,
    side_panel: bool = False,
) -> None:
    if action == "upload":
        if not local_path:
            raise ValueError("--local-path is required for action=upload")
        source = Path(local_path).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(source)
        target_path = Path(remote_path or remote_archive)
        if target_path.name == "LRDDv2.zip":
            target_path = target_path.with_name(source.name)
        with data_volume.batch_upload(force=True) as upload:
            if source.is_dir():
                upload.put_directory(source, f"/{target_path}")
            else:
                upload.put_file(source, f"/{target_path}")
        print(f"Uploaded {source} to {DATA_VOL_NAME}:/{target_path}")
        return

    if action == "ingest_lrddv2" or action == "ingest":
        print(ingest_lrddv2.remote(remote_archive=remote_archive))
        return

    if action == "train":
        print(
            train_yolo11x_p2.remote(
                data_yaml=data_yaml,
                run_name=run_name,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                resume=resume,
            )
        )
        return

    if action == "train_spawn":
        call = train_yolo11x_p2.spawn(
            data_yaml=data_yaml,
            run_name=run_name,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            resume=resume,
        )
        print(f"Spawned train_yolo11x_p2 call: {call.object_id}")
        return

    if action == "download_gdrive":
        if not file_id:
            raise ValueError("--file-id is required for action=download_gdrive")
        print(download_gdrive.remote(file_id=file_id, output_path=remote_path or "raw/download.bin"))
        return

    if action == "download_url":
        if not url:
            raise ValueError("--url is required for action=download_url")
        print(download_url.remote(url=url, output_path=remote_path or "raw/download.bin"))
        return

    if action == "download_roboflow":
        print(
            download_roboflow.remote(
                output_path=remote_path or "raw/roboflow/drones-yolo11-a.zip",
            )
        )
        return

    if action == "inspect_zip":
        print(inspect_zip.remote(remote_zip=remote_archive))
        return

    if action == "read_text":
        print(read_text.remote(remote_path=remote_path or remote_archive))
        return

    if action == "read_zip_text":
        print(read_zip_text.remote(remote_zip=remote_archive, member_suffix=remote_path or "data.yaml"))
        return

    if action == "extract_zip":
        print(extract_zip.remote(remote_zip=remote_archive, output_dir=remote_path or "raw/extracted"))
        return

    if action == "convert_detection":
        print(
            convert_detection.remote(
                remote_raw_dir=remote_archive,
                output_dir=remote_path or f"datasets/sources/{source_name}",
                source_name=source_name,
            )
        )
        return

    if action == "prepare_open_cuas":
        print(
            prepare_open_cuas.remote(
                output_dir=remote_path or "datasets/open-cuas",
                frame_stride=frame_stride,
                max_sequences=max_sequences,
            )
        )
        return

    if action == "build_validation_demo":
        print(
            build_validation_demo.remote(
                sequence_prefix=sequence_prefix,
                run_name=run_name,
                output_name=output_name,
                max_frames=max_frames,
                extra_targets=extra_targets,
                conf=conf,
                imgsz=imgsz,
                draw_scenario=draw_scenario,
                fps=fps,
            )
        )
        return

    if action == "annotate_video_demo":
        print(
            annotate_video_demo.remote(
                remote_source=remote_archive,
                run_name=run_name,
                output_name=output_name,
                conf=conf,
                imgsz=imgsz,
                iou=iou,
                side_panel=side_panel,
            )
        )
        return

    if action == "status":
        print(status.remote())
        return

    raise ValueError(f"Unknown action: {action}")
