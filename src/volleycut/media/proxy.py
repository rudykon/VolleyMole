"""Continuous CFR proxy with recorded frame selection, rotation and original audio."""
import re
import shutil
from fractions import Fraction
from pathlib import Path

from ..errors import EngineError, ErrorCode
from .probe import ffprobe_video
from .process import run_process
from .time_map import build_time_map, TIME_MAP_VERSION


def generate_proxy(src, out_path, target_fps, crf=17, keep_audio=True, *,
                   cancel_check=None, progress_cb=None, time_map_path=None):
    if shutil.which("ffmpeg") is None:
        raise EngineError(ErrorCode.INPUT_INVALID, "ffmpeg not found in PATH")
    try:
        rate = Fraction(str(target_fps))
        if rate <= 0 or rate > 240:
            raise ValueError("frame rate outside (0, 240]")
    except (ValueError, ZeroDivisionError) as exc:
        raise EngineError(ErrorCode.INPUT_INVALID, f"invalid target_fps: {target_fps}") from exc
    if Path(src).resolve() == Path(out_path).resolve() or Path(out_path).exists():
        raise EngineError(ErrorCode.INPUT_INVALID, "Proxy output must be a new file")
    info = ffprobe_video(src, cancel_check=cancel_check)
    rate_str = str(rate)
    filters = f"settb=AVTB,fps@map=fps={rate_str}:round=near:eof_action=round"
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-n", "-loglevel", "debug",
           "-copyts", "-start_at_zero", "-i", src, "-map", "0:v:0",
           "-vf", filters, "-fps_mode", "passthrough", "-c:v", "libx264",
           "-threads", "4", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
           "-map_metadata", "-1", "-metadata:s:v:0", "rotate=0", "-avoid_negative_ts", "disabled"]
    audio = next((a for a in info["audio_streams"] if a["codec"] in {"aac", "mp3", "alac"}), None)
    if keep_audio and info["has_audio"] and not audio:
        raise EngineError(ErrorCode.PROXY_FAILED, "No supported original audio stream for proxy")
    if keep_audio and audio:
        cmd += ["-map", f"0:{audio['index']}", "-c:a", "copy"]
    else:
        cmd += ["-an"]
    cmd += ["-progress", "pipe:1", "-nostats", out_path]

    def on_output(chunk):
        if progress_cb:
            matches = re.findall(r"out_time_us=(-?\d+)", chunk)
            if matches:
                progress_cb(max(0, int(matches[-1]) // 1000), int(info["duration_sec"] * 1000))

    proc = run_process(cmd, cancel_check=cancel_check, on_output=on_output)
    trace_path = str(Path(out_path).with_suffix(".ffmpeg.log"))
    Path(trace_path).write_text(proc.stderr, encoding="utf-8")
    if proc.returncode:
        raise EngineError(ErrorCode.PROXY_FAILED, proc.stderr[-2000:], {"cmd": cmd})
    proxy = ffprobe_video(out_path, cancel_check=cancel_check)
    if proxy["rotation"] % 360 or (keep_audio and info["has_audio"] and not proxy["has_audio"]):
        raise EngineError(ErrorCode.PROXY_FAILED, "Proxy rotation/audio validation failed")
    map_path = time_map_path or str(Path(out_path).with_suffix(".time_map.csv"))
    rows = build_time_map(src, out_path, map_path, fps_log=proc.stderr, fps=rate_str,
                          source_offset_ms=info["format_start_ms"], cancel_check=cancel_check)
    return {"src": src, "proxy": out_path, "target_fps": rate_str, "crf": crf,
            **{k: proxy[k] for k in ("codec", "pix_fmt", "width", "height", "nb_frames",
                                     "duration_sec", "has_audio")},
            "source_rotation": info["rotation"], "rotation_normalized": True,
            "audio_mode": "copy" if keep_audio and audio else "none",
            "audio_source_stream": audio["index"] if keep_audio and audio else None,
            "time_map": map_path, "time_map_version": TIME_MAP_VERSION,
            "mapped_frames": len(rows), "ffmpeg_trace": trace_path, "command": cmd}
