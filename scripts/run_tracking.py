import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from norfair import Detection, Tracker, draw_tracked_objects


# COCO class ids for vehicles
# 2: car, 3: motorcycle, 5: bus, 7: truck
VEHICLE_CLASS_IDS = {0, 1, 2, 3, 5, 7}


def yolo_to_norfair_detections(yolo_result, conf_thresh: float):
    """
    Convert Ultralytics YOLO result to Norfair Detection list.
    We track using bounding box center points.
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
        if k not in VEHICLE_CLASS_IDS:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="", help="Path to input video. If empty, webcam (0).")
    parser.add_argument("--out", type=str, default="outputs/out.mp4", help="Output video path.")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Ultralytics model name or path.")
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold.")
    parser.add_argument("--max_age", type=int, default=30, help="How long to keep lost tracks (frames).")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load YOLO
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

    # Output writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    # Norfair tracker
    tracker = Tracker(
        distance_function="euclidean",
        distance_threshold=60,  # tune this
        hit_counter_max=args.max_age,
    )

    frame_idx = 0
    print("Running... press 'q' to quit window preview.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # YOLO inference
        results = model.predict(frame, verbose=False)
        yres = results[0]
        detections = yolo_to_norfair_detections(yres, conf_thresh=args.conf)

        # Track update
        tracked_objects = tracker.update(detections=detections)

        # Draw tracked objects (IDs)
        vis = frame.copy()
        draw_tracked_objects(vis, tracked_objects)

        # (Optional) draw YOLO bboxes too
        for det in detections:
            x1, y1, x2, y2 = det.data["bbox"]
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(
                vis,
                f"{det.data['cls']} {float(det.scores[0]):.2f}",
                (int(x1), max(0, int(y1) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        writer.write(vis)

        cv2.imshow("CAV Multi-Object Tracking (YOLO + Norfair)", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_idx += 1

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print(f"Saved output video to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
