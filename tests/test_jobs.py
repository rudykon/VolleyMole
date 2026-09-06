import fcntl
import json
import threading
from pathlib import Path

from volleycut.jobs import JobRequest, run_detection_job

ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "experiments/stage0/samples/2.mp4"
MODEL = ROOT / "tools/fast-volleyball-tracking-inference/models/VballNetV1_seq9_grayscale_330_h288_w512.onnx"


def request(tmp_path):
    return JobRequest(str(VIDEO), str(MODEL), out_root=str(tmp_path))


def test_reject_path_traversal_before_writing(tmp_path):
    result = run_detection_job(request(tmp_path), analysis_id="../escape")
    assert result["error"]["code"] == "INPUT_INVALID"
    assert not list(tmp_path.iterdir())


def test_existing_success_cannot_be_overwritten(tmp_path):
    task = tmp_path / "used"
    task.mkdir()
    (task / "COMPLETED").write_text("keep me")
    result = run_detection_job(request(tmp_path), analysis_id="used")
    assert result["error"]["code"] == "JOB_EXISTS"
    assert (task / "COMPLETED").read_text() == "keep me"
    assert len(list(task.iterdir())) == 1


def test_cross_output_root_lock_prevents_concurrent_analysis(tmp_path):
    with open(ROOT / ".volleycut-analysis.lock", "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = run_detection_job(request(tmp_path), analysis_id="blocked")
    assert result["error"]["code"] == "JOB_BUSY"
    assert not (tmp_path / "blocked").exists()


def test_cancel_before_model_load_is_structured_and_not_complete(tmp_path):
    event = threading.Event()
    event.set()
    result = run_detection_job(request(tmp_path), cancel_event=event, analysis_id="cancelled")
    assert result["status"] == "cancelled"
    assert result["error"]["code"] == "CANCELLED"
    assert not (tmp_path / "cancelled/COMPLETED").exists()
    assert json.loads((tmp_path / "cancelled/error.json").read_text())["code"] == "CANCELLED"


def test_missing_input_is_structured_and_not_complete(tmp_path):
    req = request(tmp_path)
    req.video_path = str(tmp_path / "missing.mp4")
    result = run_detection_job(req, analysis_id="missing")
    assert result["error"]["code"] == "INPUT_INVALID"
    assert not (tmp_path / "missing/COMPLETED").exists()
