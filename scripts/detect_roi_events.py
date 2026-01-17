import argparse
import json
from pathlib import Path


def point_in_poly(x, y, poly):
    """
    Ray casting algorithm.
    poly: list of (x, y) vertices
    """
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        intersect = ((yi > y) != (yj > y)) and \
                    (x < (xj - xi) * (y - yi) / ((yj - yi) if (yj - yi) != 0 else 1e-9) + xi)
        if intersect:
            inside = not inside
        j = i
    return inside


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True, help="Path to outputs/*.tracks.json")
    ap.add_argument("--fps", type=float, default=30.0, help="FPS for converting frames->seconds")
    ap.add_argument(
        "--roi",
        required=True,
        help='ROI polygon as JSON string, e.g. \'[[100,500],[1200,500],[1200,720],[100,720]]\''
    )
    ap.add_argument("--out", default="", help="Output JSON path (optional).")
    args = ap.parse_args()

    tracks_path = Path(args.tracks)
    data = json.loads(tracks_path.read_text())

    roi = json.loads(args.roi)
    roi = [(float(p[0]), float(p[1])) for p in roi]
    if len(roi) < 3:
        raise ValueError("ROI polygon must have at least 3 points")

    events = {
        "roi": roi,
        "fps": args.fps,
        "tracks_with_events": [],
    }

    for track_id, points in data.items():
        if not points:
            continue

        # Only pedestrians
        # Note: per-point cls_id could vary; we use the most recent non-null
        cls_id = None
        for p in reversed(points):
            if p.get("cls_id") is not None:
                cls_id = p["cls_id"]
                break
        if cls_id != 0:
            continue

        # Sort by frame just in case
        points = sorted(points, key=lambda p: p["frame"])

        inside_flags = []
        for p in points:
            x, y = p["x"], p["y"]
            inside = point_in_poly(x, y, roi)
            inside_flags.append(inside)

        # Detect transitions
        entered_frame = None
        exited_frame = None

        was_inside = False
        for idx, inside in enumerate(inside_flags):
            frame = points[idx]["frame"]
            if inside and not was_inside:
                entered_frame = frame
            if (not inside) and was_inside and exited_frame is None:
                exited_frame = frame
            was_inside = inside

        # Compute time inside
        inside_frames = 0
        for idx in range(1, len(points)):
            f0 = points[idx - 1]["frame"]
            f1 = points[idx]["frame"]
            # count duration between frames if we were inside at idx-1
            if inside_flags[idx - 1]:
                inside_frames += max(0, f1 - f0)

        time_in_roi_sec = inside_frames / args.fps

        # Only keep tracks that actually entered ROI at least once
        if any(inside_flags):
            events["tracks_with_events"].append({
                "track_id": track_id,
                "entered_frame": entered_frame,
                "entered_time_sec": (entered_frame / args.fps) if entered_frame is not None else None,
                "exited_frame": exited_frame,
                "exited_time_sec": (exited_frame / args.fps) if exited_frame is not None else None,
                "time_in_roi_sec": time_in_roi_sec,
                "num_points": len(points),
            })

    out_path = Path(args.out) if args.out else tracks_path.with_suffix(".roi_events.json")
    out_path.write_text(json.dumps(events, indent=2))
    print(f"✅ Saved ROI events to: {out_path.resolve()}")
    print(f"Pedestrian tracks with ROI activity: {len(events['tracks_with_events'])}")


if __name__ == "__main__":
    main()
