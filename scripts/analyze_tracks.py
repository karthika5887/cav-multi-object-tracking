import argparse
import json
from pathlib import Path


def moving_average(values, window=5):
    if window <= 1:
        return values
    out = []
    half = window // 2
    for i in range(len(values)):
        s = 0.0
        c = 0
        for j in range(i - half, i + half + 1):
            if 0 <= j < len(values):
                s += values[j]
                c += 1
        out.append(s / c if c else values[i])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True, help="Path to *.tracks.json")
    ap.add_argument("--fps", type=float, default=30.0, help="Video FPS used for speed (px/sec).")
    ap.add_argument("--window", type=int, default=5, help="Smoothing window size.")
    ap.add_argument("--out", default="", help="Output json path (optional).")
    args = ap.parse_args()

    tracks_path = Path(args.tracks)
    data = json.loads(tracks_path.read_text())

    # JSON keys are strings -> keep as strings
    summary = {}

    for track_id, points in data.items():
        if len(points) < 2:
            continue

        # sort by frame (just in case)
        points = sorted(points, key=lambda p: p["frame"])

        frames = [p["frame"] for p in points]
        xs = [p["x"] for p in points]
        ys = [p["y"] for p in points]

        # smooth x/y to reduce jitter
        xs_s = moving_average(xs, window=args.window)
        ys_s = moving_average(ys, window=args.window)

        # compute speed in px/sec
        speeds = []
        for i in range(1, len(points)):
            dt_frames = frames[i] - frames[i - 1]
            if dt_frames <= 0:
                continue
            dx = xs_s[i] - xs_s[i - 1]
            dy = ys_s[i] - ys_s[i - 1]
            dist_px = (dx * dx + dy * dy) ** 0.5
            dt_sec = dt_frames / args.fps
            speeds.append(dist_px / dt_sec)

        cls_id = points[-1].get("cls_id", None)
        conf = points[-1].get("conf", None)

        summary[track_id] = {
            "cls_id": cls_id,
            "last_conf": conf,
            "num_points": len(points),
            "start_frame": frames[0],
            "end_frame": frames[-1],
            "duration_sec": (frames[-1] - frames[0]) / args.fps,
            "avg_speed_px_s": sum(speeds) / len(speeds) if speeds else 0.0,
            "max_speed_px_s": max(speeds) if speeds else 0.0,
        }

    out_path = Path(args.out) if args.out else tracks_path.with_suffix(".speed.json")
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"✅ Saved speed summary to: {out_path.resolve()}")
    print(f"Tracks analyzed: {len(summary)}")


if __name__ == "__main__":
    main()
