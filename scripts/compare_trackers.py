import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from norfair import Detection, Tracker, draw_tracked_objects

# COCO class IDs: 0 person, 1 bicycle, 2 car, 3 motorcycle, 5 bus, 7 truck
TRACKED_CLASS_IDS = {0, 1, 2, 3, 5, 7}


def yolo_to_norfair_detections(yolo_result, conf_thresh: float):
    dets = []
    if yolo_result.boxes is None:
        return dets

    boxes = yolo_result.boxes
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)

    for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
        if c < conf_thresh:
            continue
        if k not in TRACKED_CLASS_IDS:
            continue

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        dets.append(
            Detection(
                points=np.array([[cx, cy]], dtype=np.float32),
                scores=np.array([c], dtype=np.float32),
                data={"bbox": (float(x1), float(y1), float(x2), float(y2)), "cls": int(k)},
            )
        )
    return dets


def run_norfair(
    source: str,
    model_name: str,
    out_video: Path,
    out_json: Path,
    conf: float,
    max_age: int,
    dist_thresh: float,
):
    out_video.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_name)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1:
        fps = 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    tracker = Tracker(
        distance_function="euclidean",
        distance_threshold=dist_thresh,
        hit_counter_max=max_age,
    )

    track_history = defaultdict(list)
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        yres = model.predict(frame, verbose=False)[0]
        detections = yolo_to_norfair_detections(yres, conf_thresh=conf)
        tracked = tracker.update(detections=detections)

        # store tracks
        for obj in tracked:
            if obj.estimate is None:
                continue
            x, y = obj.estimate[0]
            det = obj.last_detection
            cls_id = det.data["cls"] if det else None
            score = float(det.scores[0]) if det and det.scores is not None else None
            bbox = det.data["bbox"] if det else None

            track_history[int(obj.id)].append(
                {
                    "frame": int(frame_idx),
                    "x": float(x),
                    "y": float(y),
                    "cls_id": int(cls_id) if cls_id is not None else None,
                    "conf": score,
                    "bbox": bbox,
                }
            )

        # draw
        vis = frame.copy()
        draw_tracked_objects(vis, tracked)

        for det in detections:
            x1, y1, x2, y2 = det.data["bbox"]
            cls_id = det.data["cls"]
            score = float(det.scores[0]) if det.scores is not None else 0.0
            color = (255, 0, 0) if cls_id == 0 else (0, 255, 0)
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(
                vis,
                f"{cls_id} {score:.2f}",
                (int(x1), max(0, int(y1) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        writer.write(vis)
        frame_idx += 1

    cap.release()
    writer.release()

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(track_history, indent=2))

    return {"fps": fps, "frames": frame_idx}


def run_bytetrack_ultralytics(source: str, model_name: str, out_video: Path, out_json: Path, conf: float):
    """
    Uses Ultralytics built-in tracker (ByteTrack).
    Exports tracks as a simplified JSON: track_id -> list of {frame, x, y, cls_id, conf, bbox}
    """
    out_video.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_name)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1:
        fps = 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    track_history = defaultdict(list)
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # persist=True tells Ultralytics to keep track IDs across frames
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=conf, verbose=False)
        r = results[0]

        vis = frame.copy()

        if r.boxes is not None and r.boxes.xyxy is not None:
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            scores = r.boxes.conf.cpu().numpy()

            # IDs may be None if tracker hasn't assigned yet
            ids = None
            if getattr(r.boxes, "id", None) is not None:
                ids = r.boxes.id.cpu().numpy().astype(int)

            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i]
                cls_id = int(cls[i])
                if cls_id not in TRACKED_CLASS_IDS:
                    continue

                score = float(scores[i])
                tid = int(ids[i]) if ids is not None else -1

                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                track_history[tid].append(
                    {
                        "frame": int(frame_idx),
                        "x": float(cx),
                        "y": float(cy),
                        "cls_id": cls_id,
                        "conf": score,
                        "bbox": (float(x1), float(y1), float(x2), float(y2)),
                    }
                )

                color = (255, 0, 0) if cls_id == 0 else (0, 255, 0)
                cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(
                    vis,
                    f"id:{tid} c:{cls_id} {score:.2f}",
                    (int(x1), max(0, int(y1) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        writer.write(vis)
        frame_idx += 1

    cap.release()
    writer.release()

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(track_history, indent=2))

    return {"fps": fps, "frames": frame_idx}


def summarize_tracks(tracks: dict):
    lengths = [len(v) for v in tracks.values() if len(v) > 0]
    return {
        "num_tracks": int(len(lengths)),
        "avg_track_len_frames": float(sum(lengths) / len(lengths)) if lengths else 0.0,
        "max_track_len_frames": int(max(lengths)) if lengths else 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="Input video path")
    ap.add_argument("--model", default="yolo11n.pt", help="YOLO model")
    ap.add_argument("--out_dir", default="outputs", help="Output directory")
    ap.add_argument("--conf", type=float, default=0.35, help="Detection confidence")
    ap.add_argument("--max_age", type=int, default=30, help="Norfair max age")
    ap.add_argument("--dist_thresh", type=float, default=60.0, help="Norfair distance threshold")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    norfair_video = out_dir / "tracked_norfair.mp4"
    norfair_json = out_dir / "tracks_norfair.json"
    bytetrack_video = out_dir / "tracked_bytetrack.mp4"
    bytetrack_json = out_dir / "tracks_bytetrack.json"
    summary_json = out_dir / "tracker_comparison_summary.json"

    print("Running Norfair...")
    run_norfair(args.source, args.model, norfair_video, norfair_json, args.conf, args.max_age, args.dist_thresh)

    print("Running ByteTrack (Ultralytics)...")
    run_bytetrack_ultralytics(args.source, args.model, bytetrack_video, bytetrack_json, args.conf)

    # Load and summarize
    norfair_tracks = json.loads(norfair_json.read_text())
    bytetrack_tracks = json.loads(bytetrack_json.read_text())

    summary = {
        "source": args.source,
        "model": args.model,
        "norfair": summarize_tracks(norfair_tracks),
        "bytetrack": summarize_tracks(bytetrack_tracks),
        "outputs": {
            "norfair_video": str(norfair_video),
            "norfair_json": str(norfair_json),
            "bytetrack_video": str(bytetrack_video),
            "bytetrack_json": str(bytetrack_json),
        },
    }

    summary_json.write_text(json.dumps(summary, indent=2))
    print(f"✅ Saved summary: {summary_json.resolve()}")
    print("Done.")


if __name__ == "__main__":
    main()
