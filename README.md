<div align="center">
  <img src="src/volleymole/assets/branding/volleymole.svg" alt="VolleyMole" width="540">
  <h1>VolleyMole · 排球高光自动剪辑</h1>
  <p>把一整场日常排球比赛，剪成有回合、有排名、有原声的竖屏五佳球或十佳球。</p>
  <p>Python 3.12 · Linux x86-64 · 单卡 / 四卡 CUDA · H.264 + AAC</p>
  <p><a href="#快速开始">快速开始</a> · <a href="#四卡加速">四卡加速</a> · <a href="#文档与开发">文档</a> · <a href="#致谢与参考">致谢</a></p>
</div>

---

## 从比赛录像到高光成片

VolleyMole 面向单机位排球录像：在本地识别比赛状态、动作、人物和球轨迹，结合多种证据切分回合，再按规则或可选的视觉大模型进行排名，输出 **720 × 1280、30 fps** 的竖屏视频。

```text
比赛录像 → 共享解码与模型分析 → 回合分割 → 排名 → 竖屏渲染 → 音画验证
                                 ↓                 ↓
                           可追溯检测证据      五佳球 / 十佳球
```

| 能力 | 说明 |
| --- | --- |
| 完整回合 | 保留检测到的回合及前后上下文，记录边界与遮挡的不确定性 |
| 跟球构图 | 球轨迹引导竖屏裁切，同时保留全场概览 |
| 活力版呈现 | 5 秒快切片头、中文倒计时排名、插画转场、短慢回放与现场原声 |
| 两种排名方式 | 纯规则模式无需 API；可选视觉大模型评审，失败时按规则降级 |
| 球员关注 | 可按球衣号码记录球员出现证据，OCR 按需启动 |
| 缓存与恢复 | 按素材、模型和配置校验分析缓存，支持从排名或渲染阶段重做 |
| 四卡流水线 | 支持跨批推理、辅助检测分卡及独立回合并行渲染 |

<p align="center">
  <img src="src/volleymole/assets/illustrated/volley_receive.png" alt="接球主题插画" width="150">
  <img src="src/volleymole/assets/illustrated/volley_set.png" alt="二传主题插画" width="150">
  <img src="src/volleymole/assets/illustrated/volley_spike.png" alt="扣球主题插画" width="150">
  <br><sub>项目内置的装饰插画；不是比赛检测结果或成片截图。</sub>
</p>

## 快速开始

**环境要求**：Linux x86-64、Python 3.12、`uv`、系统可执行的 `ffmpeg` / `ffprobe`。GPU 推理还需要兼容 CUDA 12.8 的 NVIDIA 驱动；项目不会安装或修改驱动。当前依赖锁包含 GPU 库，CPU 运行也会安装这套依赖。

### 1. 安装

```bash
git clone https://github.com/rudykon/VolleyMole.git
cd VolleyMole
uv sync --locked
.venv/bin/volleymole --help
```

实现已统一到 `src/volleymole`，无需额外检出上游项目或准备 `tools/` 文件夹。

### 2. 获取模型

```bash
.venv/bin/volleymole models --directory models fetch
.venv/bin/volleymole models --directory models verify
```

模型下载后按固定大小与 SHA-256 校验，随后可在本地推理。下载地址可能受网络限制；也支持导入已取得的对应权重，见 [模型安装说明](docs/统一包安装与运行.md)。仓库不包含模型权重或比赛录像。

### 3. 放入录像并运行

将自己的比赛录像放在 `data/match.mp4`，运行无需大模型 API 的五佳球剪辑：

```bash
.venv/bin/volleymole run \
  --video data/match.mp4 --top-k 5 --ranker rules --device cuda:0
```

默认成片：`runs/match-top5/top5_lively.mp4`。没有可用 GPU 时可改用 `--device cpu`，推理会更慢。

## 四卡加速

以下配置已在四张 RTX 3090 上完成整场实测：

```bash
.venv/bin/volleymole run \
  --video data/match.mp4 --top-k 5 --ranker rules \
  --devices cuda:0,cuda:1,cuda:2,cuda:3 \
  --pipeline-depth 2 --auxiliary-device cuda:0 \
  --vball-engine ort-bound --render-workers 2
```

四张卡分别负责状态、动作、人物和球轨迹；此命令把辅助球检测分配到 GPU 0。`--devices` 不可与 `--device` 同时使用。渲染仍用 CPU/x264；`--render-workers 2` 并行制作独立回合。

**本机单次实测**：31 分 55 秒原片 → 2 分 50 秒成片，从新鲜四卡分析到验证共 **10 分 38 秒**。检测证据与基线一致；相同呈现代码下，并行渲染与串行渲染的成片逐字节一致。此结果为规则五佳、未开启号码 OCR，不代表其他硬件或 API 模式的耗时保证。分卡与后端收益随负载变化，详见 [性能优化与验证](docs/性能优化与验证.md)。

## 常用选项

| 需求 | 参数 |
| --- | --- |
| 十佳球 | `--top-k 10` |
| 关注 12 号球员 | `--focus-player 12` |
| 经典呈现 | `--style classic`，默认 `lively` |
| 指定输出目录 | `--output runs/my-match` |
| 只分析到回合清单 | `--stop-after manifest` |
| 重做排名 / 渲染 | `--rerun-from rank` / `--rerun-from render` |
| 测量新鲜推理 | `--no-analysis-cache`，平时可复用缓存 |
| 查看全部参数 | `.venv/bin/volleymole run --help` |

## 可选：大模型排名

将 [llm_api.example.json](llm_api.example.json) 复制为本地 `llm_api.json`，填写服务地址、模型名称和 API 密钥；该本地文件已被 Git 忽略。也可配置 `VOLLEYMOLE_API_KEY`、`VOLLEYMOLE_API_BASE`、`VOLLEYMOLE_MODEL` 环境变量。

配置后用 `--ranker auto`（或省略 `--ranker`）运行。请求仅包含预筛候选的结构化信息和每回合最多三张缩小截图，**不会上传整场原视频**。接口失败或剪辑单不合法时，会记录错误类别并按规则导出。`--ranker rules` 完全关闭 API 请求。

## 输出与可追溯性

```text
runs/match-top5/
├── top5_lively.mp4           # 最终成片
├── match_manifest.json      # 回合、原片时间与检测证据
├── edit_decision.json       # 入选回合、排名及剪辑区间
├── timing_latest.json       # 本次命令分阶段耗时
├── render_report_lively.json
└── verification_lively.json # 成片解码与时间检查
```

比赛状态、回合边界和球衣号码均为模型推断，不等于人工标注真值。远景、遮挡、多球热身会影响识别，规则排名也可能选入非正式对抗。建议在发布成片前人工复核；音画验证通过不代表语义识别完全正确。

## 文档与开发

| 入口 | 内容 |
| --- | --- |
| [安装与运行](docs/统一包安装与运行.md) | 环境、模型导入、缓存、API 与运行限制 |
| [性能优化](docs/性能优化与验证.md) | 调度、后端、计时口径和复现实验 |
| [文档目录](docs/README.md) | 当前使用文档与历史开发记录 |
| [贡献指南](CONTRIBUTING.md) | 项目结构、测试与提交约定 |
| [脚本目录](scripts/README.md) | 发布检查、性能对照与本地历史审计 |

```bash
# 不需要模型下载、GPU 或真实 API 的自动测试
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'

# 上传前检查文件范围、较大文件与常见密钥格式
python3 scripts/check_release.py
```

## 致谢与参考

感谢以下项目的作者与维护者：

- [volleyball-ml-models](https://github.com/masouduut94/volleyball-ml-models) · Masoud Masoumi Moghadam：比赛状态分类、图像预处理与帧采样的实现来源。
- [fast-volleyball-tracking-inference](https://github.com/asigatchov/fast-volleyball-tracking-inference) · Alexander Sigatchov：VballNet 模型与推理参考，以及球定位、半径滤波和跟球裁切相关实现。
- [Ultralytics](https://github.com/ultralytics/ultralytics) 与 [EasyOCR](https://github.com/JaidedAI/EasyOCR)：检测与球衣号码识别基础能力。
- [PyTorch](https://pytorch.org/)、[Transformers](https://github.com/huggingface/transformers)、[ONNX Runtime](https://onnxruntime.ai/)：模型加载与推理。
- [OpenCV](https://opencv.org/)、[PyAV](https://pyav.org/)、[FFmpeg](https://ffmpeg.org/)、[NumPy](https://numpy.org/)、[SciPy](https://scipy.org/)、[Pillow](https://python-pillow.org/)：媒体处理与科学计算。
- [ZCOOL KuaiLe](https://github.com/google/fonts/tree/main/ofl/zcoolkuaile)：中文标题字体。

球衣号码功能也参考了 `volleyball-highlights` 的功能目标；当前 `jersey.py` 为独立实现，未复制该项目源码。

## 许可证与素材

第三方改编源码、依赖、字体和模型权重各自保留其许可边界，详见 [THIRD_PARTY.md](THIRD_PARTY.md) 与 [素材说明](src/volleymole/assets/README.md)。其中 Ultralytics 为 AGPL 依赖，不能把本项目整体视为 MIT 授权。

当前仓库尚未为自有代码指定项目级许可证。模型权重、比赛录像和本地凭据不随仓库分发；第三方源码许可也不自动覆盖权重、品牌标志或其他素材。
