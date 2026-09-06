import csv
import subprocess
import threading
import time
from pathlib import Path

import pytest

from volleycut.errors import EngineError, ErrorCode
from volleycut.media import ffprobe_video, generate_proxy
from volleycut.media.probe import stream_pts_ms
from volleycut.media.time_map import build_time_map

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "experiments/stage0/samples"


def audio_hash(path):
    return subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0",
                                    "-c:a", "copy", "-f", "hash", "-hash", "sha256", "-"], text=True)


@pytest.mark.real_media
@pytest.mark.parametrize("name", ["2.mp4", "IMG_0171.MOV"])
def test_real_proxy_keeps_audio_rotation_and_exact_pts(tmp_path, name):
    source = SAMPLES / name
    assert source.exists(), "Run experiments/stage0/prepare_samples.py first"
    out = tmp_path / "proxy.mp4"
    info = ffprobe_video(str(source))
    assert info["has_audio"]
    result = generate_proxy(str(source), str(out), "30")
    assert result["has_audio"] and result["rotation_normalized"]
    assert ffprobe_video(str(out))["rotation"] == 0
    assert audio_hash(source) == audio_hash(out)
    pts = stream_pts_ms(str(source))
    with open(result["time_map"], newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == result["nb_frames"]
    for row in rows:
        assert abs(float(row["source_pts_ms"]) - pts[int(row["source_frame_idx"])]) < 0.01
    assert all(abs((float(b["proxy_pts_ms"]) - float(a["proxy_pts_ms"])) - 1000 / 30) < 0.002
               for a, b in zip(rows, rows[1:]))


@pytest.mark.real_media
def test_real_ffmpeg_proxy_can_be_cancelled_promptly(tmp_path):
    started = time.monotonic()
    calls = []

    def cancel():
        calls.append(time.monotonic())
        if calls[-1] - started > 0.4:
            raise EngineError(ErrorCode.CANCELLED, "test cancellation during real media processing")

    with pytest.raises(EngineError) as raised:
        generate_proxy(str(SAMPLES / "IMG_0171.MOV"), str(tmp_path / "proxy.mp4"), 30,
                       cancel_check=cancel)
    assert raised.value.code == ErrorCode.CANCELLED
    assert time.monotonic() - started < 3
    assert len(calls) >= 3


def test_mapping_without_generation_evidence_is_rejected(tmp_path):
    with pytest.raises(EngineError) as raised:
        build_time_map("unused", "unused", str(tmp_path / "map.csv"))
    assert raised.value.code == ErrorCode.TIME_MAP_INVALID
