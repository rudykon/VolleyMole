"""Pixel-color crop suggestions ONLY. No VBallNet/tracks/rallies input, no labels written.

Every suggested point needs independent visual review. A missing suggestion never
means invisible. The native image remains authoritative, not this heuristic.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


def candidates(frame, previous):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, (15, 100, 70), (42, 255, 255))
    blue = cv2.inRange(hsv, (95, 75, 25), (135, 255, 255))
    mask = cv2.bitwise_or(yellow, blue)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray, previous) if previous is not None else np.full_like(gray, 255)
    result = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        if area < 6 or not 2 <= w <= 140 or not 2 <= h <= 140 or not .35 < w / h < 2.8:
            continue
        if not np.any(yellow[y:y+h, x:x+w]):
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        if area / (math.pi * radius * radius) < .45:
            continue
        motion = float(np.mean(diff[y:y+h, x:x+w] > 8))
        result.append({"x": float(cx), "y": float(cy), "radius": float(radius), "bbox": [x, y, w, h],
                       "area": area, "motion": motion})
    return result, gray


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("packet", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--suggestions", type=Path, help="Re-render existing unreviewed proposals; no new selection")
    parser.add_argument("--search-box", type=int, nargs=4, help="Visual navigation only: xmin ymin xmax ymax")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    with open(args.packet / "frames.csv", newline="") as stream:
        rows = list(csv.DictReader(stream))[args.start:args.stop]
    cv2.setNumThreads(4)
    existing = {e["frame_idx"]: e for e in json.loads(args.suggestions.read_text())} if args.suggestions else None
    previous, last = None, None
    proposals, panels = [], []
    for row in rows:
        image = cv2.imread(str(args.packet / row["image"]))
        choices, previous = candidates(image, previous) if existing is None else ([], None)
        if args.search_box:
            xmin, ymin, xmax, ymax = args.search_box
            choices = [c for c in choices if xmin <= c["x"] < xmax and ymin <= c["y"] < ymax]
        def rank(c):
            temporal = 1.0
            if last:
                radius = max(70, max(last["bbox"][2:]) * 5)
                distance = math.hypot(c["x"] - last["x"], c["y"] - last["y"])
                temporal = 1 + .1 / (1 + (distance / radius) ** 2)
            return math.log1p(c["area"]) * (.15 + c["motion"]) * temporal
        choices.sort(key=rank, reverse=True)
        selected = choices[0] if choices else None
        if existing is not None:
            entry = existing[int(row["frame_idx"])]
            if entry["frame_idx"] != int(row["frame_idx"]) or entry["source_pts_ms"] != row["source_pts_ms"]:
                raise ValueError("suggestion frame/PTS mismatch")
            selected = entry["suggestion"]
        if selected:
            last = selected
        proposals.append({"frame_idx": int(row["frame_idx"]), "source_pts_ms": row["source_pts_ms"],
                          "suggestion": selected, "alternatives": entry["alternatives"] if existing is not None else choices[1:4],
                          "review_status": "unreviewed_not_ground_truth"})
        panel = np.full((340, 320, 3), 245, np.uint8)
        cv2.putText(panel, f"f{row['frame_idx']}  {float(row['source_pts_ms'])/1000:.3f}s", (4, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 1)
        preview = cv2.resize(image, (320, 180))
        if selected:
            x, y = round(selected["x"]), round(selected["y"])
            cv2.rectangle(preview, (max(0, round(x / 6)-5), max(0, round(y / 6)-5)),
                          (min(319, round(x / 6)+5), min(179, round(y / 6)+5)), (0, 0, 255), 1)
            # A native 128x128 crop: no downsampling, interpolation, or hidden rescaling.
            half = 64
            left, top = max(0, min(image.shape[1] - 2*half, x-half)), max(0, min(image.shape[0] - 2*half, y-half))
            crop = image[top:top+2*half, left:left+2*half].copy()
            if "radius" in selected:
                cv2.circle(crop, (x-left, y-top), round(selected["radius"]), (0, 255, 0), 1)
            cv2.drawMarker(crop, (x-left, y-top), (0, 0, 255), cv2.MARKER_CROSS, 7, 1)
            panel[208:336, 0:128] = crop
            cv2.putText(panel, f"x={selected['x']:.1f}", (135, 235), cv2.FONT_HERSHEY_SIMPLEX, .48, (0, 0, 0), 1)
            cv2.putText(panel, f"y={selected['y']:.1f}", (135, 256), cv2.FONT_HERSHEY_SIMPLEX, .48, (0, 0, 0), 1)
            cv2.putText(panel, "SUGGESTION", (135, 283), cv2.FONT_HERSHEY_SIMPLEX, .43, (0, 0, 180), 1)
        else:
            cv2.putText(panel, "NO CROP: inspect native frame", (4, 245), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 180), 1)
        panel[24:204] = preview
        panels.append(panel)
        if len(panels) == 20 or row is rows[-1]:
            while len(panels) < 20:
                panels.append(np.full_like(panel, 245))
            sheet = np.vstack([np.hstack(panels[i:i+5]) for i in range(0, 20, 5)])
            number = (len(proposals) - 1) // 20
            cv2.imwrite(str(args.out / f"review_{number:03d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 95])
            panels = []
    (args.out / "suggestions.json").write_text(json.dumps(proposals, ensure_ascii=False, indent=2) + "\n")
    print(len(proposals), "unreviewed suggestions; no truth labels changed")


if __name__ == "__main__":
    main()
