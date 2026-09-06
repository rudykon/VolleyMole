"""结构化错误类型。失败必须可定位、不静默降级。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict


class ErrorCode(str, Enum):
    GPU_UNAVAILABLE = "GPU_UNAVAILABLE"
    MODEL_LOAD_FAILED = "MODEL_LOAD_FAILED"
    MODEL_HASH_MISMATCH = "MODEL_HASH_MISMATCH"
    VIDEO_OPEN_FAILED = "VIDEO_OPEN_FAILED"
    VIDEO_DECODE_FAILED = "VIDEO_DECODE_FAILED"
    PROXY_FAILED = "PROXY_FAILED"
    TIME_MAP_INVALID = "TIME_MAP_INVALID"
    INFERENCE_FAILED = "INFERENCE_FAILED"
    JOB_EXISTS = "JOB_EXISTS"
    JOB_BUSY = "JOB_BUSY"
    INPUT_INVALID = "INPUT_INVALID"
    IO_FAILED = "IO_FAILED"
    CANCELLED = "CANCELLED"
    INTERNAL = "INTERNAL"


@dataclass
class EngineError(Exception):
    code: ErrorCode
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.code.value}] {self.message}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": True,
            "code": self.code.value,
            "message": self.message,
            "details": self.details,
        }
