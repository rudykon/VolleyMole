# 使用指南

适用版本：VolleyMole v0.3.0；整理日期：2026-09-30。系统从单机位排球录像生成竖屏五佳球、十佳球，支持多局合并、完整回合、慢回放和模板呈现。趣味事件分析及梗配音按需启用。

`docs/` 只保留以下四份文档：

| 文档 | 内容 |
| --- | --- |
| [使用指南](README.md)（本文） | 安装、网页、剪辑、API、回放、成片与排错 |
| [自动化剪辑原理与流程](自动化剪辑原理与流程.md) | 八大步骤、本地模型、API 分工、协议与能力边界 |
| [模板与配音](模板与配音.md) | 统一模板库、设计、水墨定稿、字体素材、梗与混音 |
| [开发与验证](开发与验证.md) | 测试、性能、研究数据、评测、复现入口与清理对照 |

<a id="install"></a>

## 安装与资源准备

当前锁定环境为 Linux x86-64、Python 3.12、FFmpeg / FFprobe。GPU 依赖锁使用 PyTorch 2.11.0、torchvision 0.26.0 和 CUDA 12.8 对应发行包，需要兼容驱动；安装不修改驱动。CPU 可运行，当前没有另行精简的 CPU 发行包。

从项目根目录执行：

```bash
uv sync --locked
source .venv/bin/activate
volleymole --help
volleymole models --directory models fetch
volleymole models --directory models verify
volleymole assets fetch
volleymole assets verify
```

模型、字体和视觉素材先安装，之后本地规则剪辑可离线运行。素材固定版本为 `assets-v2`，迁移见[素材管理](模板与配音.md#assets)。使用项目锁安装，避免无锁重解依赖引入不兼容的 CUDA 库或两个 OpenCV 包。

入口脚本因环境迁移失效时，源码环境可运行 `PYTHONPATH=src .venv/bin/python -m volleymole --help`。FFmpeg / FFprobe 须在进程的 `PATH` 中。

模型目录通过 `--models` 或 `VOLLEYMOLE_MODELS` 指定；清单与哈希见 [model_manifest.json](../src/volleymole/model_manifest.json)。离线导入必须使用匹配权重，例如：

```bash
volleymole models --directory models import-file vball /path/to/VballNetV1_seq9_grayscale_330_h288_w512.onnx
```

获取失败不覆盖已有正确模型，也不自动切换未知镜像。代码和模型许可见 [THIRD_PARTY.md](../THIRD_PARTY.md)。

<a id="web"></a>

## 网页操作

```bash
volleymole web --open
volleymole web --workspace /path/to/VolleyMole --port 8787
```

默认访问 `http://127.0.0.1:8765`，不需要 Node.js 或前端构建。

1. 在“导入比赛录像”上传或选择已有源片，多局先核对日期和局号。
2. 选择五佳／十佳、竞技／趣味／双榜和模板。
3. 检查确认摘要；设备、缓存、回放、API 等在高级设置中。
4. 在任务中心查看阶段和日志，完成后播放、检查、下载。

**网页默认本地规则排名；CLI 默认 `--ranker auto`。** 网页选择趣味或双榜时切换到事件理解与 API。明确填写的选项覆盖模板，留空使用模板或命令默认值。

任务串行排队。关闭浏览器不停止任务，正常关闭服务器会取消未完成任务；重启核对进程身份，不假报成功。“任务完成”“可播放”“成片检查通过”是不同状态。

局域网示例中的 IP 应替换为实际服务器地址：

```bash
volleymole web --host 127.0.0.1 --host 172.22.13.156 --port 8787
```

服务面向可信网络内共享工作区，没有多用户登录与权限隔离。也可保持本机监听，通过 `ssh -L 8765:127.0.0.1:8765 user@server` 转发。

<a id="match"></a>

## 单视频与整场多局

```bash
# 单视频默认五佳；本例明确使用规则排名
volleymole run --video data/match.mp4 --ranker rules --template sumi --output runs/my-match
# 扫描日期，不运行模型
volleymole match --input-dir data/matches --list
# 按日期生成整场十佳球
volleymole match --input-dir data/matches --date 2026-09-29 \
  --ranker rules --template sumi --output runs/matches
```

`run` 默认五佳，`match` 默认十佳，使用 `--top-k 5` 或 `--top-k 10` 调整。可选 `--focus-player 12`；未指定号码不启动 OCR，号码识别只是筛选证据。

多局文件按 `年.月.日.局号.mp4` 命名，例如 `2026.9.29.1.mp4`、`2026.9.29.2.mp4`。递归扫描、按日期分组、按整数局号排序；重复日期与局号报错，缺失局号不补造。同日不同比赛应放入不同输入目录。

每局先分析，再合并全场候选排名，没有“每局必须入选几球”的配额。每段保留来源局与原始时间，直接从对应原片渲染。

默认输出 1080×1920、30 FPS、H.264 / AAC；`--quality` 可选 `720p`、`1080p`、`1440p`、`2160p`。完整回合路线不足指定数量的有效、不重叠回合时报错；事件路线允许不足数量输出。

<a id="api"></a>

## 规则、语义与事件模式

| 需求 | 参数 | 行为 |
| --- | --- | --- |
| 离线选球 | `--ranker rules` | 本地证据与规则评分 |
| 完整回合语义排序 | `--ranker auto --analysis-mode rallies` | 预筛后发关键帧，失败可规则兜底 |
| 竞技事件分析 | `--ranker auto --analysis-mode events --collection highlights` | 全场粗读、密集复核、程序评分 |
| 趣味榜或双榜 | `--ranker auto --collection bloopers` 或 `both` | 默认分析模式下启用事件路线 |

仅 `--ranker auto` 不启用全场事件理解；`rallies` 只支持精彩榜。规则排名配合默认回放策略不发 API 请求，显式要求 `--replay-review required` 时可能另行调用回放模型。

以 [llm_api.example.json](../llm_api.example.json) 为起点保存本地 `llm_api.json`。主流程读取 `llm` 节点的兼容服务地址、模型与密钥，也支持 `VOLLEYMOLE_API_KEY`、`VOLLEYMOLE_MODEL` 等环境变量及对应参数覆盖。密钥不放入命令行、文档或版本库。

主流程使用 OpenAI 兼容 Chat 接口，模型由配置决定。关键帧排序发送抽样图片；事件模式默认发送图片与同步音频，仅支持图片时显式使用 `--semantic-modality frames`。实际发送范围和请求记录应可核对。

事件预算默认每个源视频 1800 秒，至少 40% 留给复核，多局各有独立预算。它不包含全部本地等待、渲染与验证，也不是整个命令的硬终止时限。失败或不完整不能当作“没有精彩片段”，见[事件理解](自动化剪辑原理与流程.md#events)。

<a id="replay"></a>

## 每球慢回放

默认 2/3 倍速，`--replay-speed` 支持 0.5–1.0。

| 参数 | 结果 |
| --- | --- |
| `--replay-review auto` | 非 rules 排名且有配置时自动复核，无法确认可省略；规则排名默认不发请求，不保证每球慢放 |
| `--replay-review required` | 每球必须有合格复核记录，否则停止渲染 |
| `--replay-review off` | 不自动复核，使用有效旧记录或本地动作候选 |
| `--no-replays` | 关闭回放与自动复核，不能同时 `required` |

要求每球回放：

```bash
volleymole match --input-dir data/matches --date 2026-09-29 \
  --template sumi --replay-review required --output runs/matches-reviewed
```

需要已配置视觉模型，或与本次原片、剪辑单匹配的有效逐球复核记录。现有任务可只复核、重渲染和验收：

```bash
volleymole review-replays --run runs/my-match
python -m volleymole.media_worker render --run runs/my-match --style lively
volleymole verify --run runs/my-match --style lively
```

`required` 保证流程门槛执行，不保证模型事实判断永远准确，见[回放原理](自动化剪辑原理与流程.md#replay)。

<a id="outputs"></a>

## 文件位置、收录与迁移

| 位置 | 职责 |
| --- | --- |
| `data/` | 原片和用户素材；上传分别进入 `uploads/sources/`、`uploads/materials/` |
| `models/` | 模型权重，独立于视觉素材 |
| `templates/` | 内置、个人配置和定稿包 |
| `runs/` | 中间证据、日志、验证报告和缓存 |
| `outputs/published/` | 网页“我的成片”使用的独立成片包 |

任务常用文件：`match_manifest.json` 保存候选和来源，`edit_decision.json` 保存选段与排名，`replay_reviews.json` 保存逐球复核，`verification_lively.json` 和 `alignment_verification_lively.json` 保存技术验收。多局另有 `parts/set-0001/` 等目录及 `match_summary.json`；双榜分别写入 `collections/highlights/`、`collections/bloopers/`。

成片包固定包含 `video.mp4`、`poster.jpg`、`manifest.json`。描述文件保存标题、日期、榜单、属性、完整 SHA-256、名次摘要和验证状态，不复制源片路径、密钥或后台日志。三者齐全且描述有效才列入媒体库，完整复制该目录即可迁移。

网页成功剪辑后自动收录。CLI 任务可在网页任务报告点击“收录成片”，或运行：

```bash
python -m volleymole.web.films --workspace /path/to/VolleyMole \
  --run runs/my-match --title "周末联赛 · 十佳球"
```

收录只整理本地文件，不外部发布、不删除后台原件。已收录成片独立于 `runs/` 播放；重新分析、渲染仍需要原片和证据。任务输出不要直接指向 `outputs/published/`。

`verification=passed` 要求报告与当前成片哈希匹配；混音新文件不继承旧视频的验证状态。可播放不代表动作理解正确。完整字段以 [web/films.py](../src/volleymole/web/films.py) 为准。

<a id="cache"></a>

## 缓存、恢复与更换模板

相同命令重跑时，仅输入、参数、代码和产物校验一致的成功阶段会复用，失败阶段重试。同源五佳／十佳共享分析证据，默认缓存在 `runs/.analysis-cache/`。

```bash
# 使用原输入、原输出目录，仅更换呈现
volleymole run --video data/match.mp4 --ranker rules \
  --template aurora --output runs/my-match --rerun-from render
```

`--stop-after manifest|rank|replay|render` 分阶段处理。`--rerun-from rank|render|verify` 指定重做起点，仍检查依赖是否匹配；`--no-analysis-cache` 强制新鲜分析，`--analysis-cache-dir` 改缓存位置。

更换字体、配色通常只需重渲染；更换原片、模型、排名或复核参数会影响对应阶段。不能把缓存复用时间当作新鲜推理耗时。

<a id="troubleshooting"></a>

## 常见问题

| 问题 | 检查方式 |
| --- | --- |
| 缺字或素材缺失 | `assets verify`，部署正确素材后重渲染 |
| 模型下载失败 | 检查来源连通性，或 `import-file` 导入匹配权重 |
| 没选够十球 | 查候选排除原因，不重复片段凑数 |
| API 失败仍出了视频 | 完整回合路线可能规则兜底，查 `ranking_log.json` |
| 回放未全覆盖 | `auto` 允许省略，全覆盖需 `required` 与逐球通过记录 |
| 模板改动没影响旧片 | 对原任务重渲染与验收，旧导出文件不会自动变化 |
| GPU 不平均或编码没用 GPU | 多卡按模型分工，编码仍使用 CPU `libx264` |

完整参数以 `volleymole <命令> --help` 为准。精彩度、真实动作和梗时机的判断仍需实际观看，技术验收不能替代内容检查。
