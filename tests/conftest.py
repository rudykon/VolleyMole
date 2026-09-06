"""Public engineering fixtures; never model-quality or annotation evidence."""
import shutil
import subprocess

import pytest


@pytest.fixture(scope="session")
def synthetic_video(tmp_path_factory):
    """Exercise real preflight decoding without distributing private footage."""
    assert shutil.which("ffmpeg") and shutil.which("ffprobe"), "Install FFmpeg and FFprobe"
    path = tmp_path_factory.mktemp("synthetic-media") / "color.mp4"
    subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
        "color=c=black:s=320x240:r=12:d=1", "-an", "-c:v", "mpeg4", str(path),
    ], check=True, timeout=20)
    return path


@pytest.fixture(scope="session")
def unloaded_model(tmp_path_factory):
    """Existence-only placeholder. Tests must intercept loading; not an ONNX model."""
    path = tmp_path_factory.mktemp("unloaded-model") / "placeholder.onnx"
    path.write_bytes(b"Test placeholder: must never be decoded or used for inference.\n")
    return path
