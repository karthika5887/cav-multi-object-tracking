import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from norfair import Detection, Tracker, draw_tracked_objects


# COCO class ids:
# 0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck
TRACKED_CLASS_IDS = {0, 1, 2, 3, 5, 7}


def yolo_to_norfair_detections(yolo_result, conf_thresh: float):
    """
    Convert Ultralytics YOLO result to Norfair Detection list.
    We track using bounding box center points, but keep bbox in Detection.data.
    """
    dets = []
    if yolo_result.boxes is None:
        return dets

    boxes = yolo_result.boxes
    xyxy = boxes.xyxy.cpu().numpy()  # [N, 4]
    conf = boxes.conf.cpu().numpy()  # [N]
    cls = boxes.cls.cpu().numpy().astype(int)  # [N]

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
                data={
                    "bbox": (float(x1), float(y1), float(x2), float(y2)),
                    "cls": int(k),
                },
            )
        )

    return dets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="", help="Path to input video. If empty, webcam (0).")
    parser.add_argument("--out", type=str, default="outputs/out.mp4", help="Output video path.")
    parser.add_argument("--json_out", type=str, default="", help="Optional path to save tracks as JSON.")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Ultralytics model name or path.")
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold.")
    parser.add_argument("--max_age", type=int, default=30, help="How long to keep lost tracks (frames).")
    parser.add_argument("--dist_thresh", type=float, default=60.0, help="Norfair distance threshold.")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load YOLO detector
    model = YOLO(args.model)

    # Video input
    cap = cv2.VideoCapture(0 if args.source == "" else args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1:
        fps = 30.0

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Video output
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    # Norfair tracker
    tracker = Tracker(
        distance_function="euclidean",
        distance_threshold=args.dist_thresh,
        hit_counter_max=args.max_age,
    )

    # Track history: track_id -> list of {frame, x, y, cls_id, conf, bbox}
    track_history = defaultdict(list)

    frame_idx = 0
    print("Running... press 'q' to quit preview.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # YOLO inference
        results = model.predict(frame, verbose=False)
        yres = results[0]
        detections = yolo_to_norfair_detections(yres, conf_thresh=args.conf)

        # Update tracker
        tracked_objects = tracker.update(detections=detections)

        # Save tracks to memory (per frame)
        for obj in tracked_objects:
            if obj.estimate is None:
                continue

            x, y = obj.estimate[0]
            det = obj.last_detection
            cls_id = det.data["cls"] if det else None
            conf = float(det.scores[0]) if det and det.scores is not None else None
            bbox = det.data["bbox"] if det else None

            track_history[int(obj.id)].append(
                {
                    "frame": int(frame_idx),
                    "x": float(x),
                    "y": float(y),
                    "cls_id": int(cls_id) if cls_id is not None else None,
                    "conf": conf,
                    "bbox": bbox,
                }
            )

        # Visualization (tracks + bboxes)
        vis = frame.copy()

        # 1) draw Norfair tracks/IDs
        draw_tracked_objects(vis, tracked_objects)

        # 2) draw YOLO bboxes from detections list
        for det in detections:
            x1, y1, x2, y2 = det.data["bbox"]
            cls_id = det.data["cls"]
            score = float(det.scores[0]) if det.scores is not None else 0.0

            # Blue for person, Green for others
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

        cv2.imshow("CAV Tracking (YOLO + Norfair)", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_idx += 1

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    # Write JSON tracks
    if args.json_out:
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        json_path = out_path.with_suffix(".tracks.json")

    with open(json_path, "w") as f:
        json.dump(track_history, f, indent=2)

    print(f"Saved output video to: {out_path.resolve()}")
    print(f"Saved tracks JSON to: {json_path.resolve()}")


if __name__ == "__main__":
    main()
