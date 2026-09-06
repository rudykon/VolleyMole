"""Synthetic media preflight and deliberate error injection; no inference."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from volleycut.errors import EngineError, ErrorCode
from volleycut.jobs import JobRequest, run_detection_job
from volleycut.jobs import engine
from volleycut.perception.vballnet import DetectionConfig, VBallNetDetector
from volleycut.rallies import RallyConfig
from volleycut.tracking import TrackConfig

VIDEO = "unused.mp4"
MODEL = "unused.onnx"


@pytest.mark.parametrize("kwargs", [
    {"threshold": float("nan")}, {"threshold": float("inf")}, {"threshold": -.1},
    {"threshold": 1.1}, {"threshold": ".5"}, {"threshold": True},
    {"target_fps": 0}, {"target_fps": -1}, {"target_fps": float("nan")},
    {"target_fps": float("inf")}, {"target_fps": 241}, {"target_fps": "1/0"}, {"target_fps": True},
    {"target_fps": 30, "use_proxy": False}, {"require_cuda": "false"}, {"use_proxy": 0},
    {"model_sha256": "bad"}, {"track_config": {}},
    {"track_config": TrackConfig(roi_only="false")},
    {"rally_config": RallyConfig(spare_ball_speed_widths_per_sec=float("nan"))},
    {"rally_config": RallyConfig(head_buffer_ms=True)},
    {"rally_config": RallyConfig(max_gap_ms=250)},
    {"roi_polygon": []}, {"roi_polygon": [(0, 0), (2000, 0), (0, 100)]},
    {"roi_polygon": [(0, 0), (100, 100), (0, 100), (100, 0)]},
    {"roi_polygon": [(0, 0), (100, 0), (0, 100)], "track_config": TrackConfig(roi_only=False)},
])
def test_invalid_request_is_structured_before_model_load(tmp_path, monkeypatch, kwargs, synthetic_video, unloaded_model):
    def forbidden(*args, **kw):
        pytest.fail("invalid request attempted to construct a detector")
    monkeypatch.setattr(engine, "VBallNetDetector", forbidden)
    result = run_detection_job(JobRequest(str(synthetic_video), str(unloaded_model), out_root=str(tmp_path), **kwargs), analysis_id="invalid")
    assert result["error"]["code"] == "INPUT_INVALID"
    assert "video not found" not in result["error"]["message"]
    assert "ffmpeg/ffprobe not found" not in result["error"]["message"]
    task = tmp_path / "invalid"
    assert json.loads((task / "error.json").read_text())["code"] == "INPUT_INVALID"
    assert not (task / "COMPLETED").exists() and not (task / "detections.csv").exists()


def test_output_root_regular_file_is_structured_io_failure(tmp_path):
    root = tmp_path / "file"
    root.write_text("untouched")
    result = run_detection_job(JobRequest(str(VIDEO), str(MODEL), out_root=str(root)))
    assert result["error"]["code"] == "IO_FAILED" and root.read_text() == "untouched"


def test_progress_and_error_write_failure_does_not_escape(tmp_path, monkeypatch):
    def no_progress(*args, **kwargs):
        raise PermissionError("injected progress failure")
    def no_diagnostic(*args, **kwargs):
        raise OSError("injected full disk")
    monkeypatch.setattr(engine, "_Progress", no_progress)
    monkeypatch.setattr(engine, "_atomic_write_json", no_diagnostic)
    result = run_detection_job(JobRequest(str(VIDEO), str(MODEL), out_root=str(tmp_path)), analysis_id="io")
    assert result["error"]["code"] == "IO_FAILED"
    assert len(result["diagnostic_write_errors"]) == 2
    assert not (tmp_path / "io/COMPLETED").exists()


def test_direct_detector_rejects_bad_config_without_loading_weights():
    with pytest.raises(EngineError) as error:
        VBallNetDetector(DetectionConfig("missing.onnx", threshold=float("nan")))
    assert error.value.code is ErrorCode.INPUT_INVALID


@pytest.mark.parametrize("require_cuda", [True, False])
def test_ort_constructor_fallback_disabled_before_any_retry(monkeypatch, require_cuda, unloaded_model):
    import onnxruntime as ort
    from volleycut import runtime
    from volleycut.runtime import create_session
    monkeypatch.setattr(runtime, "preload_cuda_libs", lambda: [])
    calls = []
    def reject_constructor(*args, **kwargs):
        calls.append(kwargs)
        raise RuntimeError("deliberate constructor failure; no inference performed")
    monkeypatch.setattr(ort, "InferenceSession", reject_constructor)
    with pytest.raises(EngineError) as error:
        create_session(str(unloaded_model), require_cuda=require_cuda)
    assert len(calls) == 1 and calls[0]["enable_fallback"] is False
    assert calls[0]["providers"] == ["CUDAExecutionProvider" if require_cuda else "CPUExecutionProvider"]
    assert error.value.code is (ErrorCode.GPU_UNAVAILABLE if require_cuda else ErrorCode.MODEL_LOAD_FAILED)


def test_config_normalization_does_not_mutate_caller_in_job(tmp_path, monkeypatch, synthetic_video, unloaded_model):
    polygon = [[0, 0], [100, 0], [0, 100], [0, 0]]
    req = JobRequest(str(synthetic_video), str(unloaded_model), out_root=str(tmp_path), roi_polygon=polygon, target_fps="30000/1001")
    def stop_before_inference(config):
        req.roi_polygon[1][0] = 999
        req.threshold = .9
        assert config.threshold == .5 and config.roi_polygon[1][0] == 100
        raise EngineError(ErrorCode.CANCELLED, "test ends before inference")
    monkeypatch.setattr(engine, "VBallNetDetector", stop_before_inference)
    result = run_detection_job(req, analysis_id="snapshot")
    job = json.loads((tmp_path / "snapshot/job.json").read_text())
    assert result["status"] == "cancelled" and len(job["roi_polygon"]) == 3
    assert job["roi_polygon"][1][0] == 100 and job["threshold"] == .5
    assert len(req.roi_polygon) == 4


def test_fingerprinting_file_is_cancellable(tmp_path):
    source = tmp_path / "hash-input.bin"
    source.write_bytes(b"a" * (2 * 1024 * 1024))
    calls = 0
    def cancel():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise EngineError(ErrorCode.CANCELLED, "stop during hash")
    with pytest.raises(EngineError) as error:
        engine.video_fingerprint(str(source), cancel_check=cancel)
    assert error.value.code is ErrorCode.CANCELLED and calls == 2


def test_cli_malformed_roi_returns_json_not_traceback(tmp_path):
    proc = subprocess.run([sys.executable, "-m", "volleycut", "run", "--video", str(VIDEO),
                           "--model", str(MODEL), "--out-root", str(tmp_path), "--roi", "bad"],
                          capture_output=True, text=True, timeout=10,
                          env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")})
    assert proc.returncode == 1 and json.loads(proc.stdout)["error"]["code"] == "INPUT_INVALID"
    assert "Traceback" not in proc.stderr and not list(tmp_path.iterdir())
