"""Dense native-pixel annotation crops; no models, predictions, or label writes.

Crop locations are navigation hints only. A crop that omits the ball cannot prove
invisibility. Every final label requires visual inspection and native context.
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("packet", type=Path)
    p.add_argument("--start", type=int, required=True)
    p.add_argument("--count", type=int, required=True)
    p.add_argument("--crop", type=int, nargs=4, required=True)
    p.add_argument("--cols", type=int, default=6)
    p.add_argument("--grid", type=int, default=40)
    p.add_argument("--marks", type=Path, help="Explicit manual measuring points only, never labels")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    marks = json.loads(args.marks.read_text()) if args.marks else {}
    if args.out.exists():
        raise FileExistsError(args.out)
    with open(args.packet / "frames.csv", newline="") as stream:
        rows = list(csv.DictReader(stream))[args.start:args.start + args.count]
    x, y, w, h = args.crop
    if not rows or min(w, h, args.cols, args.grid) <= 0 or min(x, y) < 0:
        raise ValueError("invalid crop/selection")
    header = 40
    sheet = np.full((((len(rows) + args.cols - 1) // args.cols) * (h + header), args.cols * w, 3), 245, np.uint8)
    for i, row in enumerate(rows):
        source = cv2.imread(str(args.packet / row["image"]))
        crop = source[y:y+h, x:x+w].copy()
        if crop.shape[:2] != (h, w):
            raise ValueError("crop outside native frame")
        for xx in range(0, w, args.grid):
            cv2.line(crop, (xx, 0), (xx, h-1), (145, 145, 145), 1)
            cv2.putText(crop, str(x + xx), (xx+1, 11), cv2.FONT_HERSHEY_SIMPLEX, .3, (0, 0, 255), 1)
        for yy in range(0, h, args.grid):
            cv2.line(crop, (0, yy), (w-1, yy), (145, 145, 145), 1)
            cv2.putText(crop, str(y + yy), (1, yy+11), cv2.FONT_HERSHEY_SIMPLEX, .3, (0, 0, 255), 1)
        if row["frame_idx"] in marks:
            mx, my, radius = marks[row["frame_idx"]]
            cv2.circle(crop, (mx-x, my-y), radius, (0, 255, 0), 1)
            cv2.drawMarker(crop, (mx-x, my-y), (0, 0, 255), cv2.MARKER_CROSS, 7, 1)
        left, top = i % args.cols * w, i // args.cols * (h + header)
        sheet[top+header:top+header+h, left:left+w] = crop
        for j, title in enumerate((f"f{row['frame_idx']}", f"{row['source_pts_ms']} ms")):
            cv2.putText(sheet, title, (left+3, top+15+j*18), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 0), 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.out), sheet):
        raise RuntimeError("image write failed")
    print(json.dumps({"frames": [int(r["frame_idx"]) for r in rows], "native_crop": args.crop,
                      "labels_written": False, "out": str(args.out)}))


if __name__ == "__main__":
    main()
