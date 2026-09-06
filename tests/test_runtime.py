"""Backend policy control-flow tests, not evidence that CUDA inference works."""
import pytest

from volleycut import runtime
from volleycut.errors import EngineError, ErrorCode


@pytest.mark.parametrize("providers,disable_works,expected", [
    (["CUDAExecutionProvider", "CPUExecutionProvider"], True, None),
    (["CPUExecutionProvider"], True, ErrorCode.GPU_UNAVAILABLE),
    (["CUDAExecutionProvider"], False, ErrorCode.MODEL_LOAD_FAILED),
])
def test_created_session_must_confirm_cuda_and_disabled_runtime_fallback(
    monkeypatch, unloaded_model, providers, disable_works, expected,
):
    import onnxruntime as ort

    class PolicyProbe:
        _enable_fallback = True
        disable_calls = 0

        def get_providers(self):
            return providers

        def disable_fallback(self):
            self.disable_calls += 1
            if disable_works:
                self._enable_fallback = False

    session = PolicyProbe()
    calls = []

    def constructor(*args, **kwargs):
        calls.append(kwargs)
        return session

    monkeypatch.setattr(runtime, "preload_cuda_libs", lambda: [])
    monkeypatch.setattr(ort, "InferenceSession", constructor)
    if expected is None:
        assert runtime.create_session(unloaded_model) is session
    else:
        with pytest.raises(EngineError) as error:
            runtime.create_session(unloaded_model)
        assert error.value.code is expected
    assert session.disable_calls == 1
    assert len(calls) == 1 and calls[0]["enable_fallback"] is False
    assert calls[0]["providers"] == ["CUDAExecutionProvider"]


def test_missing_model_never_attempts_session_creation(monkeypatch, tmp_path):
    import onnxruntime as ort

    def forbidden(*args, **kwargs):
        pytest.fail("a missing model must fail before ORT session creation")

    monkeypatch.setattr(runtime, "preload_cuda_libs", lambda: [])
    monkeypatch.setattr(ort, "InferenceSession", forbidden)
    with pytest.raises(EngineError) as error:
        runtime.create_session(tmp_path / "missing.onnx")
    assert error.value.code is ErrorCode.MODEL_LOAD_FAILED
