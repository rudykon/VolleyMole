"""Render explicit manual measuring points for a second visual check, not labels.

No inference, coordinate interpolation, or writes to the truth packet. Each crop
is native 128 x 128 pixels. The full-frame preview is navigation context only.
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("decisions", type=Path)
    parser.add_argument("key")
    parser.add_argument("packet", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    points = json.loads(args.decisions.read_text())["packets"][args.key]["manual"]
    with (args.packet / "frames.csv").open(newline="") as stream:
        rows = {int(row["frame_idx"]): row for row in csv.DictReader(stream)}
    if len({int(p[0]) for p in points}) != len(points):
        raise ValueError("duplicate manual points")
    args.out.mkdir(parents=True, exist_ok=False)
    for offset in range(0, len(points), 20):
        sheet = np.full((4 * 340, 5 * 320, 3), 245, np.uint8)
        for index, (frame_idx, x, y) in enumerate(points[offset:offset + 20]):
            row = rows[int(frame_idx)]
            source = cv2.imread(str(args.packet / row["image"]))
            if source is None or not (0 <= x < source.shape[1] and 0 <= y < source.shape[0]):
                raise ValueError("invalid source or point")
            left = max(0, min(source.shape[1] - 128, round(x) - 64))
            top = max(0, min(source.shape[0] - 128, round(y) - 64))
            crop = source[top:top + 128, left:left + 128].copy()
            cv2.drawMarker(crop, (round(x) - left, round(y) - top), (0, 0, 255), cv2.MARKER_CROSS, 7, 1)
            ox, oy = index % 5 * 320, index // 5 * 340
            sheet[oy + 24:oy + 204, ox:ox + 320] = cv2.resize(source, (320, 180))
            sheet[oy + 208:oy + 336, ox:ox + 128] = crop
            for text, dx, dy in ((f"f{frame_idx} {row['source_pts_ms']} ms", 3, 17),
                                 (f"x={x} y={y}", 132, 235),
                                 ("MANUAL CHECK", 132, 270)):
                cv2.putText(sheet, text, (ox + dx, oy + dy), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 0), 1)
        target = args.out / f"manual_{offset // 20:03d}.png"
        if not cv2.imwrite(str(target), sheet):
            raise RuntimeError("image write failed")
    print(json.dumps({"points_rendered": len(points), "labels_written": False}))


if __name__ == "__main__":
    main()
