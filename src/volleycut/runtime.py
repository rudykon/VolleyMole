"""CUDA/ONNX Runtime 运行时：进程内预加载 cuDNN/cuBLAS，强制 CUDA EP，禁止静默回退。"""

from __future__ import annotations

import ctypes
import logging
import os
from pathlib import Path
from typing import List, Optional, Sequence

LOG = logging.getLogger(__name__)

_CUDNN_LIBS = [
    "libcudnn.so.9",
    "libcudnn_graph.so.9",
    "libcudnn_ops.so.9",
    "libcudnn_engines_precompiled.so.9",
    "libcudnn_engines_runtime_compiled.so.9",
    "libcudnn_heuristic.so.9",
    "libcudnn_cnn.so.9",
    "libcudnn_adv.so.9",
    "libcudnn_graph_backend.so.9",
]
_CUBLAS_LIBS = ["libcublasLt.so.12", "libcublas.so.12"]

_loaded: List[str] = []


def _candidate_lib_dirs() -> List[Path]:
    dirs: List[Path] = []
    try:
        import sysconfig

        sp = Path(sysconfig.get_paths()["purelib"])
        dirs.append(sp / "nvidia" / "cublas" / "lib")
        dirs.append(sp / "nvidia" / "cuda_nvrtc" / "lib")
        dirs.append(sp / "nvidia" / "cudnn" / "lib")
    except Exception:
        pass
    env = os.environ.get("VOLLEYCUT_CUDA_LIB_DIR")
    if env:
        dirs.extend(Path(p) for p in env.split(os.pathsep) if p)
    return [d for d in dirs if d.is_dir()]


def preload_cuda_libs() -> List[str]:
    """显式以 RTLD_GLOBAL 加载 cuDNN 9 / cuBLAS 12，使 ORT CUDA EP 可 dlopen 成功。

    返回成功加载的库列表；全部失败也不抛错（系统可能已有可用库），
    真实门禁由 create_session 的 CUDA EP 断言完成。
    """
    global _loaded
    if _loaded:
        return _loaded
    loaded: List[str] = []
    for lib_dir in _candidate_lib_dirs():
        libs = _CUDNN_LIBS if lib_dir.parent.name == "cudnn" else _CUBLAS_LIBS
        for name in libs:
            path = lib_dir / name
            if not path.exists():
                continue
            try:
                ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
                loaded.append(str(path))
            except OSError:
                pass
    _loaded = loaded
    if loaded:
        LOG.debug("preloaded CUDA libs: %s", loaded)
    return loaded


def create_session(
    model_path: str | os.PathLike,
    require_cuda: bool = True,
    intra_threads: Optional[int] = None,
    profile_prefix: Optional[str] = None,
) -> "object":
    """创建 ONNX Runtime 会话。

    require_cuda=True（默认）时仅注册 CUDAExecutionProvider：CUDA 不可用会直接
    抛出 EngineError(GPU_UNAVAILABLE)，绝不静默回退 CPU。
    require_cuda=False 仅在调用方显式选择 CPU 时使用（须记录于 manifest）。
    """
    from .errors import EngineError, ErrorCode

    preload_cuda_libs()
    import onnxruntime as ort

    model_path = str(model_path)
    if not os.path.exists(model_path):
        raise EngineError(
            ErrorCode.MODEL_LOAD_FAILED,
            f"model file not found: {model_path}",
            {"model_path": model_path},
        )

    opts = ort.SessionOptions()
    if profile_prefix:
        opts.enable_profiling = True
        opts.profile_file_prefix = profile_prefix
    if intra_threads:
        opts.intra_op_num_threads = intra_threads

    if require_cuda:
        providers: Sequence = ["CUDAExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    try:
        # Locked ORT 1.24.1 supports this constructor option. disable_fallback()
        # alone is too late: the constructor can already have retried on CPU.
        sess = ort.InferenceSession(model_path, sess_options=opts, providers=list(providers), enable_fallback=False)
    except Exception as exc:
        if require_cuda:
            raise EngineError(
                ErrorCode.GPU_UNAVAILABLE,
                "CUDAExecutionProvider 初始化失败；按合同不静默回退 CPU。"
                "请检查 NVIDIA 驱动、CUDA 12 运行时与 cuDNN 9（nvidia-cudnn-cu12）。",
                {"model_path": model_path, "exception": str(exc)},
            ) from exc
        raise EngineError(
            ErrorCode.MODEL_LOAD_FAILED,
            f"failed to load model: {exc}",
            {"model_path": model_path},
        ) from exc

    active = sess.get_providers()
    # Python ORT also has a run-time EPFail retry path; do not let it recreate a CPU session.
    sess.disable_fallback()
    if getattr(sess, "_enable_fallback", None) is not False:
        raise EngineError(ErrorCode.MODEL_LOAD_FAILED, "Cannot verify ORT fallback is disabled")
    if require_cuda and "CUDAExecutionProvider" not in active:
        raise EngineError(
            ErrorCode.GPU_UNAVAILABLE,
            f"CUDAExecutionProvider 未实际启用（providers={active}），拒绝继续。",
            {"model_path": model_path, "providers": active},
        )
    return sess


def describe_backend(sess) -> dict:
    import onnxruntime as ort

    return {
        "onnxruntime_version": ort.__version__,
        "providers": sess.get_providers(),
        "device": ort.get_device(),
        "provider_options": sess.get_provider_options(),
        "runtime_fallback_enabled": getattr(sess, "_enable_fallback", None),
        "constructor_fallback_enabled": False,
        "fallback_policy_version": 2,
    }
