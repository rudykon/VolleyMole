"""引擎 CLI：python -m volleycut.cli run --video ... --model ..."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading

from .errors import EngineError, ErrorCode
from .jobs import JobRequest, run_detection_job
from .tracking import TrackConfig
from .rallies import RallyConfig
from .roi import validate_polygon


def _parse_roi(s: str):
    if not s:
        return None
    pts = []
    for part in s.split(";"):
        x, _, y = part.partition(",")
        pts.append((float(x), float(y)))
    return validate_polygon(pts)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="volleycut")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="run detection job")
    run.add_argument("--video", required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--out-root", default="runs")
    run.add_argument("--target-fps", type=float, default=None)
    run.add_argument("--no-proxy", action="store_true")
    run.add_argument("--cpu-opt-in", action="store_true",
                     help="显式选择 CPU（默认强制 CUDA，失败报错不回退）")
    run.add_argument("--model-sha256", default="")
    run.add_argument("--threshold", type=float, default=0.5)
    run.add_argument("--roi", default="", help="x,y;x,y;... 分析区域多边形")
    run.add_argument("--analysis-id", default=None)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd != "run":
        return 2

    cancel_event = threading.Event()

    def _sigint(signum, frame):
        if not cancel_event.is_set():
            cancel_event.set()
            print("\n[volleycut] cancel requested, stopping after current batch...", file=sys.stderr)
        else:
            os._exit(130)

    signal.signal(signal.SIGINT, _sigint)
    signal.signal(signal.SIGTERM, _sigint)

    try:
        roi = _parse_roi(args.roi)
    except (ValueError, TypeError) as exc:
        print(json.dumps({"status": "failed", "error": EngineError(ErrorCode.INPUT_INVALID, str(exc)).to_dict()},
                         ensure_ascii=False, allow_nan=False))
        return 1
    req = JobRequest(
        video_path=args.video,
        model_path=args.model,
        out_root=args.out_root,
        target_fps=args.target_fps,
        use_proxy=not args.no_proxy,
        model_sha256=args.model_sha256,
        require_cuda=not args.cpu_opt_in,
        roi_polygon=roi,
        threshold=args.threshold,
        track_config=TrackConfig(),
        rally_config=RallyConfig(),
    )
    result = run_detection_job(req, cancel_event=cancel_event, analysis_id=args.analysis_id)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] == "completed":
        return 0
    if result["status"] == "cancelled":
        return 130
    err = result.get("error", {})
    if err.get("code") == ErrorCode.GPU_UNAVAILABLE.value:
        return 3
    return 1


if __name__ == "__main__":
    sys.exit(main())
