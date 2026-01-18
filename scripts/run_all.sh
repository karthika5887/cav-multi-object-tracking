#!/bin/bash
set -e

echo "Running Norfair tracking..."
python3 scripts/run_tracking.py \
  --source data/4KRoad_traffic_video.mp4 \
  --model yolo11n.pt \
  --out outputs/tracked.mp4 \
  --json_out outputs/tracked.tracks.json

echo "Analyzing tracks..."
python3 scripts/analyze_tracks.py \
  --tracks outputs/tracked.tracks.json \
  --fps 30

echo "Running tracker comparison..."
python3 scripts/compare_trackers.py \
  --source data/4KRoad_traffic_video.mp4 \
  --model yolo11n.pt \
  --out_dir outputs

echo "Done!"
