# Third-party source and model inventory

This document covers the integrated `src/volleymole` package. It does **not**
relicense all dependencies or model weights as MIT. No upstream GUI/server,
training pipeline, model weight or match video is included in the package.

## Incorporated source

| Module | Origin / pinned base | License | Local adaptation |
| --- | --- | --- | --- |
| `state_model.py` | `masouduut94/volleyball-ml-models`, `2374bfd859d6a03c78780d0aa1bae3d3a30dea83`, `models/GameStatusClassifierModule.py` | MIT, Copyright 2025 Masoud Masoumi Moghadam | Retains BGR/RGB preprocessing, uniform 16-frame sampling and classification. Includes local strict-checkpoint loading fix; removes logger/settings/UI dependencies; CPU uses FP32; errors propagate; probabilities retained. |
| `vball_primitives.py`, sequence handling in `tracker.py` | `asigatchov/fast-volleyball-tracking-inference`, `637c217b6589be50eba77687d3f5fa5ca103c175`, `src/inference_onnx_seq_gray_v2.py` | MIT, Copyright 2025 Alexander Sigatchov | Retains grayscale seq9 heatmap centroid and radius filtering. Includes local corrected short-tail alignment; removes independent reader/UI/pandas writes/relative CUDA paths; adds explicit device proof and confidence. |
| `camera.py` | same tracking repository, `src/make_reels.py` | same MIT | Only deployed moving-average and clamped crop paths retained; rendering/timestamps/audio remain VolleyMole code. |

Full copyright/permission notices are included in the installed package:
[`volleyball-ml-models-MIT.txt`](src/volleymole/licenses/volleyball-ml-models-MIT.txt),
[`tracking-MIT.txt`](src/volleymole/licenses/tracking-MIT.txt).
The local source files differed from the pinned upstream commits. Their exact
SHA-256 before integration:

```
GameStatusClassifierModule.py  5ba4b72cff8beb76d5bea74db117e9cd7992dd6d62ced1aae60307c4298be737
inference_onnx_seq_gray_v2.py   4c2620efea136e87bb3c0d90c1834076edf7fe49068873741de613f738936e7b
make_reels.py                  2cb5421f01bb57a16304dd11f6fa5686ab085a80e525b03483153078e81f096a
```

`jersey.py` is independently implemented from the requirement (person torso
crop → digit OCR → timestamped uncertain evidence). No source from
`volleyball-highlights` is incorporated; no root-level license was found in its
local checkout. Its private/unknown authorization is not inferred from the
availability of its source.

The analytical model loaders and batched JSON conversion are package-owned;
the analytics application's unlicensed local `run_sample.py` and
`run_batch_video.py` helpers are not imported or copied.

## Dependencies are not covered by the source MIT notices

Runtime wheels are separately resolved and hash-pinned in `uv.lock`. In particular:

- Ultralytics carries [AGPL-3.0 terms](https://github.com/ultralytics/ultralytics/blob/main/LICENSE),
  not MIT. Using a MIT adapter does not erase this dependency's obligations.
  Do not claim this combined application is permissive-only or cleared for
  closed-source redistribution. Review the intended distribution/service mode
  and applicable licensing before release.
- EasyOCR carries [Apache-2.0 terms](https://github.com/JaidedAI/EasyOCR/blob/master/LICENSE).
  VolleyMole calls the installed library; it does not copy the other project's
  number-detection implementation.
- PyTorch, Transformers, ONNX Runtime, OpenCV, PyAV, NumPy, SciPy, Pillow and the
  CUDA runtime wheels retain their respective notices in the installed distributions.
  CUDA libraries are not re-exported as VolleyMole assets.
- FFmpeg/FFprobe are external executables. The current encoder is `libx264`;
  any binary redistribution must follow the actual FFmpeg build's license.

This repository has not assigned a blanket new license to the user's
first-party code. This inventory is provenance, not a legal clearance opinion.

## Weights and artwork

`model_manifest.json` records download sources, exact byte lengths, hashes and
allowed archive members. Inference is offline and verifies hashes before load.
Only the selected files are copied to the ignored `models/` store; optimizer
states and unrelated training artifacts are excluded.

The independent ML source's MIT license does not itself establish a separate
license for the volleyball fine-tuned weights or their training data. The
current archive does not contain an explicit weight-license grant; do not
redistribute these weights as MIT. Public URLs identify provenance, not blanket
rights. The source downloader's Drive ID is used because the older README lists
a different ID. Every member and the full archive are pinned to the local
validated bytes, so an upstream replacement fails closed.

VballNet is acquired from the pinned tracking repository, person pose weights
from an Ultralytics release, and OCR weights from EasyOCR releases. No weight is
included in Git or a wheel. Availability and hash verification are tested
separately from licensing conclusions.

Artwork retains the existing asset README and generation prompts. The user's
`volleymole.svg` is included unchanged with its derived PNG; no rights in that
user-supplied logo are granted by the third-party MIT notices. ZCOOL KuaiLe font
is bundled with its OFL notice under `assets/fonts/`.
