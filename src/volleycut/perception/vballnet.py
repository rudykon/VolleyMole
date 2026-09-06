"""VBallNet seq9 连续帧检测。

预处理与解码逐步复现固定上游 637c217 的 src/inference_onnx_seq_gray_v2.py
（九帧灰度序列、288x512、/255、热力图阈值轮廓质心），输出按核心功能需求 §4
的 detections.csv 结构：frame_idx, source_pts_ms, x, y, score, model, within_roi。
"""

from __future__ import annotations

import csv
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import cv2
import numpy as np

from ..errors import EngineError, ErrorCode
from ..runtime import create_session, describe_backend
from ..models import verify_weights
from ..media.probe import stream_pts_ms
from ..roi import contains, validate_polygon
from ..validation import boolean, number

INPUT_WIDTH = 512
INPUT_HEIGHT = 288
DEFAULT_HEATMAP_THRESHOLD = 0.5


@dataclass
class DetectionConfig:
    model_path: str
    model_sha256: str = ""
    threshold: float = DEFAULT_HEATMAP_THRESHOLD
    require_cuda: bool = True
    roi_polygon: Optional[Sequence[Sequence[float]]] = None
    profile_prefix: Optional[str] = None

    def validate(self):
        number(self.threshold, "threshold", minimum=0, maximum=1)
        boolean(self.require_cuda, "require_cuda")
        return validate_polygon(self.roi_polygon)


class VBallNetDetector:
    def __init__(self, config: DetectionConfig) -> None:
        try:
            if not isinstance(config, DetectionConfig):
                raise ValueError("config must be DetectionConfig")
            self._roi = config.validate()
        except (TypeError, ValueError) as exc:
            raise EngineError(ErrorCode.INPUT_INVALID, str(exc)) from exc
        self.config = config
        self.model_sha256 = verify_weights(config.model_path, config.model_sha256)
        self.model_name = Path(config.model_path).stem
        self.session = create_session(config.model_path, require_cuda=config.require_cuda,
                                      profile_prefix=config.profile_prefix)
        inp = self.session.get_inputs()[0]
        shape = inp.shape
        if len(shape) != 4 or shape[1:] != [9, INPUT_HEIGHT, INPUT_WIDTH] or (isinstance(shape[0], int) and shape[0] != 1):
            raise EngineError(ErrorCode.MODEL_LOAD_FAILED, f"unexpected seq9 input: {shape}")
        self.input_seq, self.input_height, self.input_width = 9, INPUT_HEIGHT, INPUT_WIDTH
        self.backend = describe_backend(self.session)

    def _preprocess(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        out = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (self.input_width, self.input_height))
            out.append(gray.astype(np.float32) / 255.0)
        return out

    def _decode(self, output: np.ndarray) -> List[tuple]:
        results = []
        out_dim = output.shape[1]
        for i in range(out_dim):
            heatmap = output[0, i, :, :]
            _, binary = cv2.threshold(heatmap, self.config.threshold, 1.0, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(
                (binary * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                results.append((0, 0, 0, 0.0))
                continue
            largest = max(contours, key=cv2.contourArea)
            moments = cv2.moments(largest)
            if moments["m00"] == 0:
                results.append((0, 0, 0, 0.0))
                continue
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            score = float(heatmap[cy, cx])
            results.append((1, cx, cy, score))
        return results

    def _within_roi(self, x: int, y: int) -> bool:
        return contains(self._roi, (float(x), float(y)))

    def detect_file(
        self,
        video_path: str,
        source_pts_ms: Optional[List[float]] = None,
        output_csv: Optional[str] = None,
        progress_cb: Optional[Callable[[int, int], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> List[dict]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            raise EngineError(
                ErrorCode.VIDEO_OPEN_FAILED, f"could not open video: {video_path}"
            )
        # Raw inputs are engineering diagnostics; use the same rotated pixel axes.
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
        if cap.get(cv2.CAP_PROP_ORIENTATION_META) % 360 and not cap.get(cv2.CAP_PROP_ORIENTATION_AUTO):
            cap.release()
            raise EngineError(ErrorCode.VIDEO_DECODE_FAILED, "Decoder cannot normalize source rotation; use a proxy")
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_width <= 0 or frame_height <= 0:
            cap.release()
            raise EngineError(ErrorCode.VIDEO_DECODE_FAILED, f"bad frame size: {video_path}")

        try:
            self._roi = validate_polygon(self._roi, frame_width, frame_height)
            if source_pts_ms is None:
                source_pts_ms = stream_pts_ms(video_path, kind="frame")
        except ValueError as exc:
            cap.release()
            raise EngineError(ErrorCode.INPUT_INVALID, str(exc)) from exc
        except BaseException:
            cap.release()
            raise
        if not source_pts_ms:
            cap.release()
            raise EngineError(ErrorCode.TIME_MAP_INVALID, "No decoded source frame PTS")
        batch = self.input_seq
        frame_buffer = []
        rows: List[dict] = []
        frame_index = 0
        csv_file = None
        csv_writer = None
        try:
            if output_csv:
                Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
                csv_file = open(output_csv, "w", newline="", encoding="utf-8")
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(
                    ["frame_idx", "source_pts_ms", "x", "y", "score", "model", "within_roi"]
                )
        except BaseException:
            cap.release()
            if csv_file:
                csv_file.close()
            raise

        def pts_of(idx: int) -> float:
            if source_pts_ms is not None and idx < len(source_pts_ms):
                return round(source_pts_ms[idx], 3)
            raise EngineError(ErrorCode.TIME_MAP_INVALID, f"No source PTS for frame {idx}")

        try:
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    raise EngineError(ErrorCode.CANCELLED, "detection cancelled by request")
                frames = []
                for _ in range(batch):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame.shape[:2] != (frame_height, frame_width):
                        raise EngineError(ErrorCode.VIDEO_DECODE_FAILED, "Decoded dimensions differ from coordinate space")
                    frames.append(frame)
                if not frames:
                    break

                processed = self._preprocess(frames)
                # Preserve even the upstream's partial-tail behaviour: prepend history,
                # keep the last nine inputs, emit the FIRST len(frames) outputs.
                # Its tail alignment is a known baseline limitation, not silently fixed.
                while len(frame_buffer) < batch:
                    frame_buffer.append(processed[0])
                frame_buffer.extend(processed)
                frame_buffer = frame_buffer[-batch:]

                tensor = np.stack(frame_buffer, axis=2)[np.newaxis, ...]
                tensor = np.transpose(tensor, (0, 3, 1, 2)).astype(np.float32, copy=False)
                try:
                    outputs = self.session.run(None, {self.session.get_inputs()[0].name: tensor})
                except Exception as exc:
                    raise EngineError(ErrorCode.INFERENCE_FAILED, str(exc),
                                      {"frame_idx": frame_index}) from exc
                preds = self._decode(outputs[0])

                for i, (vis, x, y, score) in enumerate(preds[: len(frames)]):
                    idx = frame_index + i
                    if vis:
                        xo = int(x * frame_width / self.input_width)
                        yo = int(y * frame_height / self.input_height)
                        within = self._within_roi(xo, yo)
                    else:
                        xo, yo, within = -1, -1, False
                    row = {
                        "frame_idx": idx,
                        "source_pts_ms": pts_of(idx),
                        "x": xo,
                        "y": yo,
                        "score": round(float(score), 6),
                        "model": self.model_name,
                        "within_roi": bool(within),
                    }
                    rows.append(row)
                    if csv_writer is not None:
                        csv_writer.writerow(
                            [
                                row["frame_idx"],
                                f"{row['source_pts_ms']:.3f}",
                                row["x"],
                                row["y"],
                                f"{row['score']:.6f}",
                                row["model"],
                                int(row["within_roi"]),
                            ]
                        )

                frame_index += len(frames)
                if progress_cb is not None:
                    progress_cb(frame_index, total_frames if total_frames > 0 else frame_index)
            if frame_index != len(source_pts_ms):
                raise EngineError(ErrorCode.VIDEO_DECODE_FAILED,
                                  "Decoded frames do not cover the complete PTS mapping",
                                  {"decoded": frame_index, "expected": len(source_pts_ms)})
        finally:
            cap.release()
            if csv_file is not None:
                csv_file.close()
        return rows
