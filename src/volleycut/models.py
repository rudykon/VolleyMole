"""Weights frozen by docs/阶段0-1/模型登记.md; code license is not weight clearance."""

import hashlib
from pathlib import Path

from .errors import EngineError, ErrorCode

UPSTREAM_COMMIT = "637c217b6589be50eba77687d3f5fa5ca103c175"
WEIGHTS = {
    "VballNetV1_seq9_grayscale_330_h288_w512.onnx":
        "2f36bd129c51a4d8afb9a23fd4816be578c65041d463d55a81d1667735070b76",
    "VballNetV1_seq9_grayscale_148_h288_w512.onnx":
        "81ee27fe68d3e9b1e991e3d17ef529af85708609c482d3b6397bfbd9becda884",
}


def verify_weights(path: str, expected: str = "") -> str:
    try:
        with open(path, "rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
    except OSError as exc:
        raise EngineError(ErrorCode.MODEL_LOAD_FAILED, str(exc)) from exc
    frozen = WEIGHTS.get(Path(path).name)
    if not frozen or actual != frozen or (expected and actual != expected.lower()):
        raise EngineError(ErrorCode.MODEL_HASH_MISMATCH, "权重不符合阶段 0–1 冻结清单。",
                          {"expected": expected or frozen, "actual": actual})
    return actual
