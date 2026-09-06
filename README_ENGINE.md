# 阶段 0/1 本地分析引擎

这是 VolleyMole 的独立阶段 0/1 实验引擎，Python 模块名仍为 `volleycut`。
本次同步保留远端已有 React 界面和 `services/inference` 服务骨架，**尚未把本引擎接入它们**。
上传代码不表示产品已交付或阶段门禁已通过。

## 当前状态

- 独立真值标注尚未完成，疑难回合结束事件仍保留待定。
- 验证集冻结、质量基线、验收配置、用户实测节时四项门禁全部未通过。
- 代码与单元测试不构成模型质量或用户节时的验收证据。

详情：[数据与复现说明](docs/阶段0-1/数据与复现说明.md)、
[模型与许可登记](docs/阶段0-1/模型登记.md)。

## 引擎运行

与现有服务的 Python 环境独立。本引擎需要 Python 3.12、FFmpeg/FFprobe，以及可用的
NVIDIA CUDA 环境；依赖版本见根目录 `pyproject.toml` 和 `uv.lock`。

```bash
uv sync --python 3.12
PYTHONPATH=src uv run python -m volleycut --help
PYTHONPATH=src uv run python -m volleycut run \
  --video /absolute/path/to/match.mp4 \
  --model /absolute/path/to/authorized-model.onnx \
  --out-root runs
```

默认要求 CUDA，构造与运行时的静默 CPU 回退均禁用。应自行取得有权使用的模型，并核对
模型登记中的哈希；本仓库不分发模型权重，也未确认权重的独立分发许可。

## 可公开复现的检查

公开工程测试不读取用户视频、实际标注或实验报告；使用临时生成的色块视频和校验器夹具，
不做模型推理、不生成质量基线。需要 FFmpeg/FFprobe。完整材料测试需另行准备本地数据。

```bash
PYTHONPATH=src uv run --locked pytest -q -m 'not local_data'
```

GitHub Actions 在推送和 PR 时运行上述完整公开集合，不再只挑选五个测试文件。
测试分层、轻量 CI 环境和本地全量命令见 [回归测试与 CI](docs/阶段0-1/回归测试与CI.md)。
`local_data` 只控制显式测试选择；直接运行全套测试时，缺少本地材料仍会失败，不会静默跳过。

完整测试、真实视频验证、图片哈希复核和历史 CUDA 实验还需要本地源视频、模型、原尺寸
图片、上游固定提交及实验产物。它们没有随本次 GitHub 同步上传；不要把克隆后的缺失材料
误认为校验通过。JSON、CSV、XML 报告及本地实验记录也不上传。

整个 `data/` 目录不上传，包括 CSV/JSON 初标、材料清单和源视频指纹。原始视频、逐帧 PNG、
审阅图片、音频、实验日志与报告、缓存、源快照 ZIP、完整第三方工具目录和令牌文件均留在本地。
公开文档仅说明格式、操作流程与当前未通过的门禁，不提供逐帧标签或实际回合边界。

## 第三方许可

抽取或参考的上游代码与修改说明见
[第三方来源与修改说明](docs/阶段0-1/第三方来源与修改说明.md)。所需 MIT 版权与许可全文
保留在 `tools/fast-volleyball-tracking-inference/LICENSE` 和
`tools/basketball-shot-clipper/LICENSE`。代码许可证不等于权重授权。
