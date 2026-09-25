<div align="center">
  <img src="src/volleymole/assets/branding/volleymole.svg" alt="VolleyMole" width="400">
  <h1>VolleyMole · 排球视频自动剪辑</h1>
  <p>从单机位比赛录像生成竖屏五佳球、十佳球，保留完整回合与现场原声。</p>
</div>

支持多局合并排名、跟球裁切、慢回放、中英标题和五套视觉设计。默认输出 **1080 × 1920 / 30 fps / H.264 + AAC**，可选 720p、1440p、2160p。分析结果支持缓存与断点恢复；事件双榜和梗配音可单独启用。

## 快速开始

环境：Linux x86-64、Python 3.12、[uv](https://docs.astral.sh/uv/)、FFmpeg / FFprobe。GPU 运行需要兼容 CUDA 12.8 的驱动；当前依赖锁包含 GPU 库。

```bash
git clone https://github.com/rudykon/VolleyMole.git
cd VolleyMole
uv sync --locked
source .venv/bin/activate

# 下载并校验模型与视觉素材，之后可离线使用
volleymole models --directory models fetch
volleymole assets fetch

# 单视频五佳球；规则排名无需 API
volleymole run --video data/match.mp4 --ranker rules \
  --template matchday --output runs/my-match
```

整场多局录像按 `年.月.日.局号.mp4` 命名，例如 `2026.9.15.1.mp4`：

```bash
volleymole match --input-dir data/matches --list
volleymole match --input-dir data/matches --date 2026-09-15 \
  --ranker rules --template atelier --output runs/matches
```

`run` 默认五佳，`match` 默认十佳；使用 `--top-k 5` 或 `--top-k 10` 调整。输出目录保存成片、剪辑单和验证报告。更多安装说明见[安装与运行](docs/统一包安装与运行.md)。

## 网页工作台

安装依赖后启动中文网页界面：

```bash
volleymole web --open
# 默认 http://127.0.0.1:8765；可用 --port 更换端口
# 同时允许局域网设备访问
volleymole web --host 127.0.0.1 --host 172.22.13.156 --port 8787
```

网页支持录像上传、单视频与多局剪辑、全部公开参数、模板编辑、任务队列与取消、真实视频封面与预览下载、慢回放复核、配音时间点编辑、模型与素材安装，以及 API 与算法配置。创建页按录像、目标、风格排列并提供可读摘要；首页优先展示任务与最近成片，媒体库分别管理比赛源片、剪辑素材与正式成片。`runs/` 仅供后台处理；网页成片统一保存在 `outputs/published/`，每份包含 `video.mp4`、`poster.jpg`、`manifest.json`，清理后台不影响已收录成片。任务详情显示真实阶段记录，生成后可直接播放，并将执行完成与成片检查分开展示。无需 Node.js 或前端构建，详见[网页工作台](docs/网页工作台.md)与[成片目录及格式](docs/成片目录与格式.md)。

## 自己的成片模板

模板使用 JSON 保存风格、语言、转场、画质、慢回放及配音设置，可以直接读取文件，也可以导入后按名称切换。

```bash
# 查看五套内置模板
volleymole templates list

# 导出后按自己的习惯编辑
volleymole templates export matchday --name my-team --output my-team.json

# 导入本地 templates/，随后按名称使用
volleymole templates import my-team.json
volleymole run --video data/match.mp4 --ranker rules --template my-team

# 也可直接使用 JSON，临时覆盖语言或画质
volleymole run --video data/match.mp4 --ranker rules \
  --template ./my-team.json --design-language en --quality 1440p
```

| 模板 | 视觉设计 |
| --- | --- |
| `matchday` | 赤线竞技 |
| `atelier` | 纸上球场 |
| `sumi` | 墨间回合 |
| `aurora` | 极光棱镜 |
| `archive` | 胶片纪事 |

<p>
  <img src="docs/images/design-suites/matchday.png" width="150" alt="赤线竞技">
  <img src="docs/images/design-suites/atelier.png" width="150" alt="纸上球场">
  <img src="docs/images/design-suites/aurora.png" width="150" alt="极光棱镜">
</p>

使用 `design_suite: "custom"` 可自由组合插画、标题和转场。配音仍是成片后的独立步骤，同一份模板通过 `meme-audio --template my-team` 生效。完整格式、优先级和命令见[自定义成片模板](docs/自定义成片模板.md)。

## 常用选项

| 需求 | 参数 / 入口 |
| --- | --- |
| 纯本地规则排名 | `--ranker rules` |
| 语义排名 | 复制 `llm_api.example.json` 为 `llm_api.json`，配置 API 后使用 `--ranker auto` |
| 竞技与趣味双榜 | `--collection both --analysis-mode events --ranker auto` |
| 指定单卡 | `--device cuda:0` |
| 四卡并行 | `--devices cuda:0,cuda:1,cuda:2,cuda:3 --pipeline-depth 2` |
| 只重做呈现 | 原命令保留输出目录，加 `--rerun-from render` |
| 要求慢回放全部复核通过 | `--replay-review required`，需要已配置视觉 API |
| 添加本地配音 | `volleymole meme-audio --help` |
| 查看全部参数 | `volleymole run --help` / `volleymole match --help` |

语义模式会向配置的服务发送抽样画面，部分事件模式还使用音频。规则排名可离线运行；语义排名失败会记录规则兜底。检测、动作判断与自动配音仍可能出错，发布前请检查成片。详见[事件双榜](docs/事件理解与双榜.md)、[慢回放](docs/慢回放选点与完整性.md)与[配音](docs/克制梗配音.md)。

## 仓库内容

```text
src/volleymole/     运行代码、内置模板、提示词与素材来源
scripts/           素材准备、评测、预览与发布检查
tests/            回归测试
docs/             使用指南与设计预览
```

视频、模型、密钥、个人模板、运行缓存、实验报告和本地归档均不提交。完整插画和字体由独立素材包分发；GitHub 保留少量预览图。

```bash
python scripts/check_release.py --history  # 文件、历史凭据及大文件检查
python scripts/check_release.py --archive dist/VolleyMole-github.zip  # 导出干净源码
python -m unittest discover -s tests       # 完整测试需先安装视觉素材
```

开发说明见[CONTRIBUTING](CONTRIBUTING.md)，更多指南见[文档目录](docs/README.md)。项目代码采用 [MIT](LICENSE)；依赖、改编代码和模型的许可分别见[第三方来源](THIRD_PARTY.md)。
