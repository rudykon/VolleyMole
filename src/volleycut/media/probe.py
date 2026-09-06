"""ffprobe 封装：只读探测，真实时间戳，不依赖 CAP_PROP_FPS。"""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import Dict, List

from ..errors import EngineError, ErrorCode
from .process import run_process


def _run_ffprobe(args: List[str], cancel_check=None) -> str:
    if shutil.which("ffprobe") is None:
        raise EngineError(ErrorCode.INPUT_INVALID, "ffprobe not found in PATH")
    proc = run_process(["ffprobe", "-hide_banner", *args], cancel_check=cancel_check)
    if proc.returncode != 0:
        raise EngineError(
            ErrorCode.VIDEO_OPEN_FAILED,
            f"ffprobe failed: {proc.stderr.strip()[:500]}",
            {"args": args},
        )
    return proc.stdout


def ffprobe_video(path: str, cancel_check=None) -> Dict:
    out = _run_ffprobe(
        [
            "-v", "error",
            "-show_streams", "-show_format",
            "-of", "json",
            path,
        ], cancel_check=cancel_check
    )
    data = json.loads(out)
    streams = data.get("streams") or []
    videos = [s for s in streams if s.get("codec_type") == "video"]
    if not videos:
        raise EngineError(
            ErrorCode.VIDEO_OPEN_FAILED, f"no video stream found: {path}", {"path": path}
        )
    s = videos[0]
    fmt = data.get("format", {})

    def _frac(v: str) -> float:
        try:
            num, _, den = str(v).partition("/")
            den_f = float(den) if den else 1.0
            return float(num) / den_f if den_f else 0.0
        except ValueError:
            return 0.0

    return {
        "path": path,
        "codec": s.get("codec_name"),
        "pix_fmt": s.get("pix_fmt"),
        "width": int(s.get("width", 0)),
        "height": int(s.get("height", 0)),
        "avg_frame_rate": _frac(s.get("avg_frame_rate", "0/1")),
        "avg_frame_rate_ratio": s.get("avg_frame_rate", "0/1"),
        "r_frame_rate_ratio": s.get("r_frame_rate", "0/1"),
        "time_base": s.get("time_base"),
        "start_pts_ms": float(s.get("start_time") or 0) * 1000,
        "format_start_ms": float(fmt.get("start_time") or 0) * 1000,
        "rotation": next((int(d["rotation"]) for d in s.get("side_data_list", [])
                          if "rotation" in d), int(s.get("tags", {}).get("rotate", 0))),
        "audio_streams": [{"index": a["index"], "codec": a.get("codec_name"),
                           "start_time": a.get("start_time"), "duration": a.get("duration")}
                          for a in streams if a.get("codec_type") == "audio"],
        "r_frame_rate": _frac(s.get("r_frame_rate", "0/1")),
        "nb_frames": int(s.get("nb_frames") or 0),
        "duration_sec": float(s.get("duration") or fmt.get("duration") or 0.0),
        "has_audio": any(st.get("codec_type") == "audio" for st in streams),
        "format_name": fmt.get("format_name"),
    }


def stream_pts_ms(path: str, kind: str = "frame", cancel_check=None) -> List[float]:
    """按流顺序读取视频流时戳（毫秒）。kind=packet 用包时戳（源视频真值），
    kind=frame 用解码帧时戳（代理视频）。"""
    if kind == "packet":
        entry = "packet=pts_time"
        flag = "-show_packets"
    else:
        entry = "frame=best_effort_timestamp_time"
        flag = "-show_frames"
    out = _run_ffprobe(
        [
            "-v", "error",
            "-select_streams", "v:0",
            flag,
            "-show_entries", entry,
            "-of", "csv=p=0",
            path,
        ], cancel_check=cancel_check
    )
    pts: List[float] = []
    for line in out.splitlines():
        val = line.split(",")[0].strip()
        try:
            pts.append(float(val) * 1000.0)
        except ValueError:
            continue
    return pts
