"""最小检测引擎任务编排：进度、取消、任务隔离、结构化错误、可复现输出。

任务目录布局（runs/<analysis_id>/）：
  job.json progress.jsonl proxy.mp4 time_map.csv detections.csv
  tracks.jsonl rallies.json run_manifest.json job_result.json
  COMPLETED（成功原子标志）/ CANCEL（外部取消请求）/ error.json（失败）
"""

from __future__ import annotations

import hashlib
import csv
import fcntl
import json
import os
import re
import subprocess
import shutil
import threading
import time
import uuid
import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .. import __version__
from ..errors import EngineError, ErrorCode
from ..media import build_time_map, ffprobe_video, generate_proxy
from ..media.probe import stream_pts_ms
from ..media.time_map import load_time_map, write_time_map, TIME_MAP_VERSION
from ..models import UPSTREAM_COMMIT
from ..perception import DetectionConfig, VBallNetDetector
from ..rallies import QUALITY_RULE_VERSION, RallyConfig, generate_rallies
from ..runtime import describe_backend
from ..tracking import TrackConfig, build_tracks, write_tracks_jsonl
from ..tracking.clean import RULE_VERSION as TRACK_RULE_VERSION
from ..roi import ROI_RULE_VERSION, validate_polygon
from ..validation import boolean, frame_rate, number

FINGERPRINT_BYTES = 16 * 1024 * 1024
CANCEL_FILE = "CANCEL"
COMPLETED_FILE = "COMPLETED"


@dataclass
class JobRequest:
    video_path: str
    model_path: str
    out_root: str = "runs"
    target_fps: Optional[float] = None
    use_proxy: bool = True
    model_sha256: str = ""
    require_cuda: bool = True
    roi_polygon: Optional[Sequence[Sequence[float]]] = None
    threshold: float = 0.5
    track_config: TrackConfig = field(default_factory=TrackConfig)
    rally_config: RallyConfig = field(default_factory=RallyConfig)

    def validate(self):
        for name in ("video_path", "model_path", "out_root"):
            value = os.fspath(getattr(self, name))
            if not isinstance(value, str) or not value.strip() or "\x00" in value:
                raise ValueError(f"{name} must be a nonempty local path")
            setattr(self, name, value)
        if Path(self.video_path).suffix.lower() not in {".mp4", ".mov"}:
            raise ValueError("video_path must be MP4 or MOV")
        boolean(self.use_proxy, "use_proxy")
        boolean(self.require_cuda, "require_cuda")
        number(self.threshold, "threshold", minimum=0, maximum=1)
        if self.target_fps is not None:
            self.target_fps = str(frame_rate(self.target_fps))
            if not self.use_proxy:
                raise ValueError("target_fps requires a proxy; raw diagnostic input keeps its native PTS")
        if not isinstance(self.model_sha256, str) or (self.model_sha256 and not re.fullmatch(r"[a-fA-F0-9]{64}", self.model_sha256)):
            raise ValueError("model_sha256 must be empty or a 64-character SHA-256")
        if not isinstance(self.track_config, TrackConfig) or not isinstance(self.rally_config, RallyConfig):
            raise ValueError("track_config/rally_config must be their respective config dataclasses")
        self.track_config.validate()
        self.rally_config.validate()
        if self.track_config.max_gap_ms != self.rally_config.max_gap_ms:
            raise ValueError("track and rally max_gap_ms must agree for interpretable gap flags")
        self.roi_polygon = validate_polygon(self.roi_polygon)
        if self.roi_polygon is not None and not self.track_config.roi_only:
            raise ValueError("an explicit analysis ROI cannot be silently disabled by roi_only=False")


def _sha256_file(path: str, limit: Optional[int] = None, cancel_check=None) -> str:
    h = hashlib.sha256()
    read = 0
    with open(path, "rb") as f:
        while True:
            if cancel_check:
                cancel_check()
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
            read += len(chunk)
            if limit is not None and read >= limit:
                break
    return h.hexdigest()


def video_fingerprint(path: str, cancel_check=None) -> str:
    size = os.path.getsize(path)
    return f"sha256:{_sha256_file(path, cancel_check=cancel_check)}:{size}"


def _engine_source_hash() -> str:
    root = Path(__file__).resolve().parent.parent
    h = hashlib.sha256()
    for py in sorted(root.rglob("*.py")):
        h.update(py.relative_to(root).as_posix().encode())
        h.update(py.read_bytes())
    return h.hexdigest()


def _atomic_write_text(path: str, text: str) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def _atomic_write_json(path: str, obj) -> None:
    _atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


class _Progress:
    def __init__(self, path: str) -> None:
        self.path = path
        self._fh = open(path, "a", encoding="utf-8")

    def emit(self, stage: str, done: int = 0, total: int = 0, **extra) -> None:
        rec = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "stage": stage,
            "done": done,
            "total": total,
        }
        rec.update(extra)
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def _check_cancel(task_dir: Path, cancel_event: Optional[threading.Event]) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise EngineError(ErrorCode.CANCELLED, "cancelled by event")
    if (task_dir / CANCEL_FILE).exists():
        raise EngineError(ErrorCode.CANCELLED, f"cancelled by {CANCEL_FILE} file")


def run_detection_job(
    req: JobRequest,
    cancel_event: Optional[threading.Event] = None,
    analysis_id: Optional[str] = None,
) -> Dict:
    """One active analysis per workspace, new task directories only; errors never overwrite."""
    analysis_id = uuid.uuid4().hex if analysis_id is None else analysis_id
    if not isinstance(analysis_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", analysis_id):
        return {"analysis_id": analysis_id if isinstance(analysis_id, str) else None, "status": "failed", "error":
                EngineError(ErrorCode.INPUT_INVALID, "Invalid analysis_id").to_dict()}
    def failed(code, message, details=None):
        return {"analysis_id": analysis_id, "status": "failed", "error":
                EngineError(code, message, details or {}).to_dict()}
    try:
        if not isinstance(req, JobRequest):
            return failed(ErrorCode.INPUT_INVALID, "req must be JobRequest")
        # Freeze caller-owned mutable config/ROI for this invocation.
        req = copy.deepcopy(req)
        if not os.fspath(req.out_root).strip() or "\x00" in os.fspath(req.out_root):
            return failed(ErrorCode.INPUT_INVALID, "out_root must be a nonempty local path")
        root = Path(req.out_root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        task_dir = root / analysis_id
        # One shared lock regardless of the requested output root.
        lock_path = Path(__file__).resolve().parents[3] / ".volleycut-analysis.lock"
        with open(lock_path, "a+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return failed(ErrorCode.JOB_BUSY, "An analysis is already running")
            try:
                task_dir.mkdir(exist_ok=False)
            except FileExistsError:
                return failed(ErrorCode.JOB_EXISTS, "Task directory already exists")
            return _run_detection_job(req, task_dir, cancel_event, analysis_id)
    except (ValueError, TypeError) as exc:
        return failed(ErrorCode.INPUT_INVALID, str(exc))
    except OSError as exc:
        return failed(ErrorCode.IO_FAILED, str(exc), {"type": type(exc).__name__, "errno": exc.errno})


def _run_detection_job(
    req: JobRequest,
    task_dir: Path,
    cancel_event: Optional[threading.Event] = None,
    analysis_id: Optional[str] = None,
) -> Dict:

    progress = None
    result: Dict = {
        "analysis_id": analysis_id,
        "task_dir": str(task_dir),
        "status": "failed",
    }
    error_obj: Optional[Dict] = None

    def fail(exc: EngineError) -> Dict:
        nonlocal error_obj
        error_obj = exc.to_dict()
        result["status"] = "cancelled" if exc.code is ErrorCode.CANCELLED else "failed"
        result["error"] = error_obj
        return result

    cancel_check = lambda: _check_cancel(task_dir, cancel_event)

    try:
        progress = _Progress(str(task_dir / "progress.jsonl"))
        try:
            req.validate()
        except (TypeError, ValueError) as exc:
            raise EngineError(ErrorCode.INPUT_INVALID, str(exc)) from exc
        video_path = os.path.abspath(req.video_path)
        model_path = os.path.abspath(req.model_path)
        if not os.path.isfile(video_path):
            raise EngineError(
                ErrorCode.INPUT_INVALID, f"video not found: {video_path}"
            )
        if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
            raise EngineError(ErrorCode.INPUT_INVALID, "ffmpeg/ffprobe not found in PATH")

        _atomic_write_json(
            str(task_dir / "job.json"),
            {
                "analysis_id": analysis_id,
                "created_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                "video_path": video_path,
                "model_path": model_path,
                "target_fps": req.target_fps,
                "use_proxy": req.use_proxy,
                "require_cuda": req.require_cuda,
                "threshold": req.threshold,
                "roi_polygon": [list(p) for p in req.roi_polygon] if req.roi_polygon else None,
                "track_config": vars(req.track_config), "rally_config": vars(req.rally_config),
            },
        )

        progress.emit("precheck")
        _check_cancel(task_dir, cancel_event)

        src_info = ffprobe_video(video_path, cancel_check=cancel_check)
        if src_info["width"] <= 0 or src_info["height"] <= 0:
            raise EngineError(ErrorCode.VIDEO_DECODE_FAILED, "invalid source video size")
        expected_width, expected_height = src_info["width"], src_info["height"]
        if not req.use_proxy and src_info["rotation"] % 90:
            raise EngineError(ErrorCode.INPUT_INVALID, "Raw diagnostic input requires right-angle rotation; use a proxy")
        if src_info["rotation"] % 180:
            expected_width, expected_height = expected_height, expected_width
        # Rectilinear rotation dimensions are known without running a model.
        if src_info["rotation"] % 90 == 0:
            try:
                validate_polygon(req.roi_polygon, expected_width, expected_height)
            except ValueError as exc:
                raise EngineError(ErrorCode.INPUT_INVALID, str(exc)) from exc

        detector = VBallNetDetector(
            DetectionConfig(
                model_path=model_path,
                model_sha256=req.model_sha256,
                threshold=req.threshold,
                require_cuda=req.require_cuda,
                roi_polygon=req.roi_polygon,
                profile_prefix=str(task_dir / "ort_profile"),
            )
        )
        backend = detector.backend
        fingerprint = video_fingerprint(video_path, cancel_check=cancel_check)

        if req.use_proxy:
            progress.emit("proxy")
            _check_cancel(task_dir, cancel_event)
            target_fps = req.target_fps or src_info["r_frame_rate_ratio"]
            proxy_info = generate_proxy(
                video_path, str(task_dir / "proxy.mp4"), target_fps=target_fps,
                time_map_path=str(task_dir / "time_map.csv"), cancel_check=cancel_check,
                progress_cb=lambda done, total: progress.emit("proxy", done, total),
            )
            detect_video = str(task_dir / "proxy.mp4")
            source_pts_ms = load_time_map(str(task_dir / "time_map.csv"))
            from fractions import Fraction
            detect_fps = float(Fraction(str(target_fps)))
            detect_width = proxy_info["width"]
            detect_height = proxy_info["height"]
            duration_ms = src_info["start_pts_ms"] + src_info["duration_sec"] * 1000
        else:
            proxy_info = None
            detect_video = video_path
            source_pts_ms = stream_pts_ms(video_path, cancel_check=cancel_check)
            write_time_map([{"frame_idx": i, "proxy_pts_ms": pts,
                             "source_frame_idx": i, "source_pts_ms": pts}
                            for i, pts in enumerate(source_pts_ms)], str(task_dir / "time_map.csv"))
            detect_fps = src_info["avg_frame_rate"] or src_info["r_frame_rate"]
            detect_width = expected_width
            detect_height = expected_height
            duration_ms = src_info["start_pts_ms"] + src_info["duration_sec"] * 1000.0

        try:
            validate_polygon(req.roi_polygon, detect_width, detect_height)
            number(detect_fps, "decoded frame rate", positive=True)
        except ValueError as exc:
            raise EngineError(ErrorCode.INPUT_INVALID, str(exc)) from exc
        progress.emit("detect")
        last_emit = [0.0]

        def on_progress(done: int, total: int) -> None:
            _check_cancel(task_dir, cancel_event)
            now = time.monotonic()
            if now - last_emit[0] >= 0.5 or done >= total:
                progress.emit("detect", done, total)
                last_emit[0] = now

        detections = detector.detect_file(
            detect_video,
            source_pts_ms=source_pts_ms,
            output_csv=str(task_dir / "detections.csv"),
            progress_cb=on_progress,
            cancel_event=cancel_event,
        )
        profile_path = detector.session.end_profiling()
        profile = json.loads(Path(profile_path).read_text())
        kernels = [p for p in profile if p.get("cat") == "Node" and "provider" in p.get("args", {})]
        provider_counts = {}
        for kernel in kernels:
            provider = kernel["args"]["provider"]
            provider_counts[provider] = provider_counts.get(provider, 0) + 1
        if req.require_cuda and not provider_counts.get("CUDAExecutionProvider"):
            raise EngineError(ErrorCode.GPU_UNAVAILABLE, "No CUDA kernels in real inference profile")
        backend["executed_node_counts"] = provider_counts
        backend["profile"] = Path(profile_path).name

        progress.emit("tracks")
        _check_cancel(task_dir, cancel_event)
        track_decisions = []
        tracks = build_tracks(
            detections,
            fps=detect_fps,
            frame_width=detect_width,
            config=req.track_config,
            roi_polygon=req.roi_polygon,
            frame_height=detect_height,
            decisions=track_decisions,
            cancel_check=cancel_check,
        )
        write_tracks_jsonl(tracks, str(task_dir / "tracks.jsonl"))
        write_tracks_jsonl(track_decisions, str(task_dir / "track_decisions.jsonl"))

        progress.emit("rallies")
        _check_cancel(task_dir, cancel_event)
        rallies_doc = generate_rallies(
            tracks,
            video_duration_ms=duration_ms,
            analysis_id=analysis_id,
            video_fingerprint=fingerprint,
            frame_width=detect_width,
            frame_height=detect_height,
            config=req.rally_config,
            video_start_ms=src_info["start_pts_ms"],
            roi_polygon=req.roi_polygon,
            cleaning_decisions=track_decisions,
        )
        _atomic_write_json(str(task_dir / "rallies.json"), rallies_doc)
        _atomic_write_text(str(task_dir / "edits.jsonl"), "")

        progress.emit("finalize")
        manifest = {
            "schema_version": 1,
            "analysis_id": analysis_id,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "volleycut_version": __version__,
            "engine_source_sha256": _engine_source_hash(),
            "code_commit": _code_commit(),
            "upstream_commit": UPSTREAM_COMMIT,
            "source_video": {
                "path": video_path,
                "fingerprint": fingerprint,
                "codec": src_info["codec"],
                "pix_fmt": src_info["pix_fmt"],
                "width": src_info["width"],
                "height": src_info["height"],
                "rotation": src_info["rotation"],
                "start_pts_ms": src_info["start_pts_ms"],
                "format_start_ms": src_info["format_start_ms"],
                "avg_frame_rate": src_info["avg_frame_rate"],
                "duration_sec": src_info["duration_sec"],
            },
            "rotation_normalized": True,
            "roi": {"polygon": req.roi_polygon, "coordinate_space": "rotation_normalized_input_pixels",
                    "width": detect_width, "height": detect_height, "rule_version": ROI_RULE_VERSION},
            "proxy": proxy_info,
            "time_map": {"path": "time_map.csv", "sha256": _sha256_file(str(task_dir / "time_map.csv")),
                         "version": TIME_MAP_VERSION, "frames": len(source_pts_ms),
                         "method": "ffmpeg_fps_trace" if req.use_proxy else "decoded_frame_pts"},
            "model": {
                "path": model_path,
                "sha256": _sha256_file(model_path),
                "name": Path(model_path).stem,
            },
            "inference_backend": backend,
            "require_cuda": req.require_cuda,
            "thresholds": {
                "heatmap_threshold": req.threshold,
                "track": vars(req.track_config),
                "rally": vars(req.rally_config),
            },
            "quality_flags_rule_version": QUALITY_RULE_VERSION,
            "tracking_rule_version": TRACK_RULE_VERSION,
            "counts": {
                "frames_decoded": len(detections),
                "tracks": len(tracks),
                "rallies": len(rallies_doc["rallies"]),
            },
        }
        manifest["output_sha256"] = {name: _sha256_file(str(task_dir / name))
                                     for name in ("detections.csv", "tracks.jsonl", "track_decisions.jsonl", "rallies.json", "edits.jsonl")}
        cancel_check()
        _atomic_write_json(str(task_dir / "run_manifest.json"), manifest)

        _atomic_write_json(
            str(task_dir / "job_result.json"),
            {"analysis_id": analysis_id, "status": "completed"},
        )
        progress.emit("done", 1, 1)
        _atomic_write_text(str(task_dir / COMPLETED_FILE), manifest["created_utc"] + "\n")

        result["status"] = "completed"
        result["counts"] = manifest["counts"]
        result["backend"] = backend
        return result

    except EngineError as exc:
        return fail(exc)
    except OSError as exc:
        return fail(EngineError(ErrorCode.IO_FAILED, str(exc), {"type": type(exc).__name__, "errno": exc.errno}))
    except Exception as exc:
        return fail(
            EngineError(ErrorCode.INTERNAL, f"unexpected error: {exc}", {"type": type(exc).__name__})
        )
    finally:
        # A full/unwritable disk must not hide the primary structured failure.
        def record_safely(action):
            try:
                action()
            except OSError as exc:
                result.setdefault("diagnostic_write_errors", []).append(str(exc))
        if error_obj is not None:
            record_safely(lambda: _atomic_write_json(str(task_dir / "error.json"), error_obj))
            record_safely(lambda: _atomic_write_json(str(task_dir / "job_result.json"),
                          {"analysis_id": analysis_id, "status": result["status"], "error": error_obj}))
            if progress is not None:
                record_safely(lambda: progress.emit("error", error=error_obj["code"]))
        if progress is not None:
            record_safely(progress.close)


def _code_commit():
    proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[3],
                          capture_output=True, text=True)
    return proc.stdout.strip() if proc.returncode == 0 else None
