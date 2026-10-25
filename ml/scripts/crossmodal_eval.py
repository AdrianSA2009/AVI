"""Cross-modal evaluation: YOLO image->video, plus video->image preparation.

Modes:
  image-to-video: stream each alphabet video through YOLO; report frame +
    video majority-vote metrics, sliding-window-smoothed metrics, latency,
    FPS, confusion matrices, and hard examples.
  prepare-video-to-image: sample video frames, reject blur/near-duplicates,
    pseudo-label with YOLO, then group-split by source video. Stops before
    training: review/correct labels before training (explicit requirement).

Only alphabet A-I,K-Y compared: detector excludes dynamic J/Z. Video frames
inherit folder label. This is class evaluation only; no bounding-box mAP since
video ground truth has no per-frame boxes.

Run from C:\\AVI\\ml:
  python scripts\\crossmodal_eval.py --mode image-to-video
  python scripts\\crossmodal_eval.py --mode prepare-video-to-image
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import yaml
import torch
from ultralytics import YOLO

DEFAULT_MODEL = Path("runs/detect/runs/detect/fold_1/weights/best.pt")
DEFAULT_VIDEO = Path("data/video_test/alphabet")
DEFAULT_IMAGE_FOLDS = Path("datasets/sibi/folds.json")
DYNAMIC = {"J", "Z"}
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def load_names() -> dict[int, str]:
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "datasets/sibi/run1.yaml").read_text(encoding="utf-8"))
    return {int(k): str(v).upper() for k, v in cfg["names"].items()}


def label_from_path(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    return rel.parts[0].upper() if len(rel.parts) > 1 else ""


def frame_metrics(records: list[dict], labels: list[str]) -> dict:
    ids = {label: i for i, label in enumerate(labels)}
    valid = [r for r in records if r["truth"] in ids]
    # Include NO_DETECTION as explicit prediction column.
    cols = labels + ["NO_DETECTION"]
    cm = [[0 for _ in cols] for _ in labels]
    col_ids = {label: i for i, label in enumerate(cols)}
    for r in valid:
        cm[ids[r["truth"]]][col_ids.get(r["pred"], col_ids["NO_DETECTION"])] += 1
    correct = sum(cm[i][i] for i in range(len(labels)))
    per_class = {}
    for i, c in enumerate(labels):
        tp, support = cm[i][i], sum(cm[i])
        predicted = sum(cm[row][i] for row in range(len(labels)))
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[c] = {"support": support, "precision": precision,
                        "recall": recall, "f1": f1}
    present = [m for m in per_class.values() if m["support"]]
    n = len(valid)
    return {"samples": n, "accuracy": correct / n if n else None,
            "macro_precision": sum(x["precision"] for x in present) / len(present) if present else None,
            "macro_recall": sum(x["recall"] for x in present) / len(present) if present else None,
            "macro_f1": sum(x["f1"] for x in present) / len(present) if present else None,
            "per_class": per_class, "matrix_labels": cols, "confusion_matrix": cm}


def video_vote(frame_rows: list[dict], labels: list[str], smooth: int) -> tuple[dict, dict]:
    by_video: dict[str, list[dict]] = defaultdict(list)
    for row in frame_rows:
        by_video[row["video"]].append(row)
    raw, smoothed = [], []
    for video, rows in by_video.items():
        truth = rows[0]["truth"]
        votes = Counter(r["pred"] for r in rows if r["pred"] != "NO_DETECTION")
        raw_pred = votes.most_common(1)[0][0] if votes else "NO_DETECTION"
        raw.append({"video": video, "truth": truth, "pred": raw_pred})
        history: deque[str] = deque(maxlen=smooth)
        smoothed_rows = []
        for row in rows:
            history.append(row["pred"])
            window_votes = Counter(x for x in history if x != "NO_DETECTION")
            frame_pred = window_votes.most_common(1)[0][0] if window_votes else "NO_DETECTION"
            smoothed_rows.append(frame_pred)
        votes_smoothed = Counter(x for x in smoothed_rows if x != "NO_DETECTION")
        smooth_pred = votes_smoothed.most_common(1)[0][0] if votes_smoothed else "NO_DETECTION"
        smoothed.append({"video": video, "truth": truth, "pred": smooth_pred})
    return frame_metrics(raw, labels), frame_metrics(smoothed, labels)


def run_image_to_video(args: argparse.Namespace) -> dict:
    model_path = args.weights
    model = YOLO(str(model_path))
    names = load_names()
    labels = list(names.values())
    videos = sorted(p for p in args.source.rglob("*")
                    if p.is_file() and p.suffix.lower() in VIDEO_EXT)
    if not videos:
        raise SystemExit(f"Tidak ada video di {args.source}")
    rows: list[dict] = []
    latencies: dict[str, list[float]] = defaultdict(list)
    error_dir = args.out / "errors"
    error_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    error_saved = 0
    for vi, video in enumerate(videos, start=1):
        truth = label_from_path(video, args.source)
        if truth in DYNAMIC or truth not in labels:
            continue
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            continue
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 0
        cap.release()
        frame_index = 0
        for result in model.predict(source=str(video), device=0, stream=True,
                                    conf=args.conf, verbose=False):
            timing = result.speed or {}
            for key in ("preprocess", "inference", "postprocess"):
                if key in timing:
                    latencies[key].append(float(timing[key]))
            boxes = result.boxes
            if boxes is not None and len(boxes):
                best_i = int(boxes.conf.argmax())
                pred = names.get(int(boxes.cls[best_i]), "NO_DETECTION")
                confidence = float(boxes.conf[best_i])
            else:
                pred, confidence = "NO_DETECTION", 0.0
            rows.append({"video": str(video), "frame": frame_index,
                         "truth": truth, "pred": pred, "confidence": confidence,
                         "source_fps": fps_src})
            if pred != truth and error_saved < args.error_samples:
                # Ultralytics result.orig_img is sampled frame; save limited examples.
                if rng.random() < 0.25:
                    safe = video.stem.replace(" ", "_")
                    cv2.imwrite(str(error_dir / f"{safe}_f{frame_index}_{truth}_{pred}.jpg"),
                                result.orig_img)
                    error_saved += 1
            frame_index += 1
        if vi % 25 == 0:
            print(f"Video {vi}/{len(videos)}", flush=True)

    frame_raw = frame_metrics(rows, labels)
    smooth_frames: list[dict] = []
    for video in sorted({r["video"] for r in rows}):
        subset = [r for r in rows if r["video"] == video]
        q: deque[str] = deque(maxlen=args.smooth)
        for row in subset:
            q.append(row["pred"])
            valid = Counter(x for x in q if x != "NO_DETECTION")
            smoothed = valid.most_common(1)[0][0] if valid else "NO_DETECTION"
            smooth_frames.append({**row, "pred": smoothed})
    frame_smooth = frame_metrics(smooth_frames, labels)
    video_raw, video_smooth = video_vote(rows, labels, args.smooth)
    n = len(rows)
    latency = {k: {"mean_ms": float(np.mean(v)) if v else None,
                   "p95_ms": float(np.percentile(v, 95)) if v else None}
               for k, v in latencies.items()}
    infer = latencies.get("inference", [])
    report = {
        "mode": "image-to-video", "weights": str(model_path),
        "video_root": str(args.source), "frame_samples": n,
        "videos_evaluated": len({r["video"] for r in rows}),
        "excluded_dynamic_classes": sorted(DYNAMIC), "conf": args.conf,
        "smoothing_window_frames": args.smooth,
        "frame_metrics_raw": frame_raw, "frame_metrics_smoothed": frame_smooth,
        "video_metrics_majority": video_raw, "video_metrics_smoothed": video_smooth,
        "latency_ultralytics_ms_per_frame": latency,
        "fps_from_inference_only": 1000 / float(np.mean(infer)) if infer and np.mean(infer) else None,
        "note": "Frame-level class scoring only: video GT provides clip class, "
                "not frame boxes. No mAP because no bbox annotations. FPS inverse "
                "inference speed excludes decode/network/display overhead.",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "image_to_video_frames.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["video", "frame", "truth", "pred", "confidence", "source_fps"])
        writer.writeheader()
        writer.writerows(rows)
    metric_rows = []
    for level, metrics in (("frame_raw", frame_raw), ("frame_smoothed", frame_smooth),
                           ("video_majority", video_raw), ("video_smoothed", video_smooth)):
        for cls, values in metrics["per_class"].items():
            metric_rows.append({"level": level, "class": cls, **values})
    with (args.out / "image_to_video_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["level", "class", "support", "precision", "recall", "f1"])
        writer.writeheader()
        writer.writerows(metric_rows)
    (args.out / "image_to_video.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    save_confusion(frame_raw, args.out / "image_to_video_confusion.png", "Image -> video frame")
    save_comparison(report, args.out / "image_to_video_comparison.png")
    return report


def save_comparison(report: dict, out: Path) -> None:
    import matplotlib.pyplot as plt
    levels = ["frame_metrics_raw", "frame_metrics_smoothed",
              "video_metrics_majority", "video_metrics_smoothed"]
    names = ["Frame", "Frame + smoothing", "Video vote", "Video smoothed"]
    metrics = [report[k] for k in levels]
    keys = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
    x = np.arange(len(keys))
    width = 0.19
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (name, item) in enumerate(zip(names, metrics)):
        ax.bar(x + (i - 1.5) * width, [item[k] or 0 for k in keys], width, label=name)
    ax.set_xticks(x, ["Accuracy", "Macro P", "Macro R", "Macro F1"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Cross-domain YOLO: image -> video")
    ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def save_confusion(metrics: dict, out: Path, title: str) -> None:
    import matplotlib.pyplot as plt
    labels = metrics["matrix_labels"]
    cm = np.asarray(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(13, 11))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=90)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Prediksi"); ax.set_ylabel("Label benar"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def prepare_video_to_image(args: argparse.Namespace) -> dict:
    """Extract clear frames, remove within-video near duplicates, YOLO pseudo-label.

    Group split is performed by source video, never frame. Output requires
    human review before YOLO training; function does not train.
    """
    model = YOLO(str(args.weights))
    names = load_names()
    videos = sorted(p for p in args.source.rglob("*")
                    if p.is_file() and p.suffix.lower() in VIDEO_EXT)
    out_images = args.out / "video_to_image" / "images"
    out_labels = args.out / "video_to_image" / "labels"
    out_images.mkdir(parents=True, exist_ok=True); out_labels.mkdir(parents=True, exist_ok=True)
    candidates: list[dict] = []
    seen_hashes: dict[str, list[int]] = defaultdict(list)
    excluded = Counter()
    sample_every = args.sample_every
    rng = random.Random(args.seed)

    for vi, video in enumerate(videos):
        truth = label_from_path(video, args.source)
        if truth in DYNAMIC or truth not in names.values():
            excluded[truth] += 1
            continue
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            continue
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % sample_every:
                index += 1; continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            if blur < args.blur_threshold:
                index += 1; continue
            tiny = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
            dh = 0
            for yy in range(8):
                for xx in range(8):
                    dh = (dh << 1) | int(tiny[yy, xx] > tiny[yy, xx + 1])
            # compare with retained hashes in this same video only
            if any((dh ^ prev).bit_count() <= args.dhash_distance
                   for prev in seen_hashes[str(video)]):
                index += 1; continue
            results = model.predict(frame, device=0, conf=args.conf,
                                    verbose=False, imgsz=640)[0]
            if results.boxes is None or len(results.boxes) == 0:
                index += 1; continue
            stem = f"v{vi:04d}_f{index:06d}"
            image_path = out_images / f"{stem}.jpg"
            label_path = out_labels / f"{stem}.txt"
            cv2.imwrite(str(image_path), frame)
            yolo_lines = []
            source_class_id = next(i for i, name in names.items() if name == truth)
            for box, conf in zip(results.boxes.xywhn.cpu().tolist(),
                                 results.boxes.conf.cpu().tolist()):
                if conf < args.conf:
                    continue
                yolo_lines.append(" ".join([str(source_class_id), *(f"{v:.6f}" for v in box)]))
            label_path.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")
            candidates.append({"image": str(image_path), "label": str(label_path),
                               "video": str(video), "frame": index,
                               "truth": truth, "blur_laplacian": blur,
                               "pseudo_classes": ",".join(sorted({names[int(b.cls[0])]
                                   for b in results.boxes}))})
            seen_hashes[str(video)].append(dh)
            index += 1
        cap.release()
        if vi and vi % 25 == 0:
            print(f"Video {vi}/{len(videos)}", flush=True)

    # group by source video, deterministic split .70/.15/.15, no video leakage
    group_labels: dict[str, str] = {}
    for item in candidates:
        group_labels[item["video"]] = item["truth"]
    groups_by_class: dict[str, list[str]] = defaultdict(list)
    for video, cls in group_labels.items(): groups_by_class[cls].append(video)
    split_video: dict[str, str] = {}
    for cls, groups in groups_by_class.items():
        rng.shuffle(groups)
        n = len(groups)
        if n < 3:
            splits = ["train"] * n
        else:
            nval = max(1, round(n * .15)); ntest = max(1, round(n * .15))
            if nval + ntest >= n: nval, ntest = 1, 1
            splits = ["test"] * ntest + ["val"] * nval + ["train"] * (n - ntest - nval)
        split_video.update(dict(zip(groups, splits)))
    # Add explicit group assignment to review manifest; each frame follows source video.
    for item in candidates:
        item["split"] = split_video.get(item["video"], "train")

    # Generate YOLO data.yaml file lists; labels need manual review first.
    list_paths = {}
    for split in ("train", "val", "test"):
        list_path = args.out / "video_to_image" / f"{split}.txt"
        selected = [i["image"] for i in candidates if split_video.get(i["video"]) == split]
        list_path.write_text("\n".join(selected) + ("\n" if selected else ""), encoding="utf-8")
        list_paths[split] = str(list_path)
    yaml_data = {"path": str((args.out / "video_to_image").resolve().as_posix()),
                 "train": list_paths["train"], "val": list_paths["val"],
                 "test": list_paths["test"], "names": {i: v for i, v in names.items()}}
    (args.out / "video_to_image" / "data.yaml").write_text(
        yaml.safe_dump(yaml_data, sort_keys=False), encoding="utf-8")
    with (args.out / "video_to_image" / "review.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "label", "video", "frame", "truth", "split", "blur_laplacian", "pseudo_classes"])
        writer.writeheader()
        writer.writerows(candidates)
    report = {"frames_pseudo_labeled": len(candidates), "videos_with_frames": len(group_labels),
              "video_split_counts": dict(Counter(split_video.values())),
              "excluded_classes": dict(excluded), "data_yaml": str(args.out / "video_to_image" / "data.yaml"),
              "review_csv": str(args.out / "video_to_image" / "review.csv"),
              "must_review": True, "training_started": False,
              "caveat": "Pseudo boxes must be checked/corrected manually before training. "
                        "Group split per source video prevents frame-level leakage."}
    (args.out / "video_to_image" / "preparation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("image-to-video", "prepare-video-to-image", "all"), default="all")
    parser.add_argument("--weights", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("outputs/crossmodal"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--smooth", type=int, default=7)
    parser.add_argument("--error-samples", type=int, default=100)
    parser.add_argument("--sample-every", type=int, default=8)
    parser.add_argument("--blur-threshold", type=float, default=30.0)
    parser.add_argument("--dhash-distance", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not args.weights.is_file(): raise SystemExit(f"Weights tidak ada: {args.weights}")
    if not torch.cuda.is_available(): raise SystemExit("CUDA tidak aktif")
    print(f"GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__}")
    if args.source is None: args.source = DEFAULT_VIDEO
    image_report = video_prep_report = None
    if args.mode in ("image-to-video", "all"):
        image_report = run_image_to_video(args)
    if args.mode in ("prepare-video-to-image", "all"):
        video_prep_report = prepare_video_to_image(args)
    # Preserve results from earlier individual mode runs.
    image_path = args.out / "image_to_video.json"
    prep_path = args.out / "video_to_image" / "preparation.json"
    if image_report is None and image_path.is_file():
        image_report = json.loads(image_path.read_text(encoding="utf-8"))
    if video_prep_report is None and prep_path.is_file():
        video_prep_report = json.loads(prep_path.read_text(encoding="utf-8"))

    (args.out / "reports").mkdir(parents=True, exist_ok=True)
    summary = {"image_to_video": image_report,
               "video_to_image_preparation": video_prep_report,
               "in_domain_image_baseline": "runs/metrics_cv.json",
               "prior_in_domain_video_training_only_loss": "runs/video_sibi_alphabet_test/report.json",
               "domain_gap": "Compare same-class, same-label metrics. Existing source data "
                             "has one signer/one clip per class; label/domain differences "
                             "and video frame sampling limit conclusions."}
    (args.out / "reports" / "crossmodal_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Evaluasi lintas-modalitas", ""]
    if image_report:
        lines += ["## YOLO gambar -> video", "",
                  f"- Checkpoint: `{image_report['weights']}`",
                  f"- Video: {image_report['videos_evaluated']}; frame: {image_report['frame_samples']}; J/Z dikecualikan.",
                  f"- Akurasi frame: {image_report['frame_metrics_raw']['accuracy']:.4f}; "
                  f"macro F1: {image_report['frame_metrics_raw']['macro_f1']:.4f}.",
                  f"- Video majority: {image_report['video_metrics_majority']['accuracy']:.4f}; "
                  f"smoothed: {image_report['video_metrics_smoothed']['accuracy']:.4f}.",
                  f"- Inference: {image_report['latency_ultralytics_ms_per_frame'].get('inference', {}).get('mean_ms')} ms/frame; "
                  f"~{image_report['fps_from_inference_only']:.1f} FPS inference-only.",
                  "- mAP tidak dihitung: video tidak punya bounding-box GT per frame.", ""]
    if video_prep_report:
        lines += ["## Video -> gambar (persiapan pseudo-label)", "",
                  f"- Frame kandidat: {video_prep_report['frames_pseudo_labeled']}; "
                  f"video berframe: {video_prep_report['videos_with_frames']}.",
                  f"- Split video: `{video_prep_report['video_split_counts']}`.",
                  "- Label pseudo wajib ditinjau/dikoreksi manual. Training belum dijalankan.",
                  f"- Data YAML: `{video_prep_report['data_yaml']}`.", ""]
    lines += ["## Domain gap dan batasan", "",
              "Model gambar diuji lintas-sumber pada video; hasilnya bukan bukti kualitas "
              "deployment realtime. Hasil video->gambar belum tersedia karena pseudo-label "
              "menunggu review manual. Gunakan split per video dan jangan klaim generalisasi "
              "dari training loss.", ""]
    (args.out / "reports" / "crossmodal_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Summary: {args.out / 'reports' / 'crossmodal_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())