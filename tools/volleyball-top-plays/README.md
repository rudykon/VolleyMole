# VolleyMole 第一版

输入固定或轻微抖动机位的 MP4 / MOV / AVI，自动分析整场，输出 720×1280、30 fps、H.264/AAC 的五佳或十佳。画面采用跟球细节与全场小画面，保留现场声，按第 K 名至第 1 名倒计时播放。没有原声的输入补静音轨以保持拼接一致。

默认 `--style lively` 使用橙黄、薄荷绿、珊瑚粉标题与入场动效；4 秒片头由五段各 0.8 秒的精彩镜头快切组成。每个完整回合后，重播已评审动作关键帧附近约 2 秒的源画面，以 2/3 速播放约 3 秒，音频同步降速而不改变音高。回合间插入 0.4 秒斜向擦除连接动画及轻提示音，不裁去回合动作。可用 `--style classic` 生成第一版样式；两种成片、片段和报告分别保存。

活泼标题只改表达方式，不添加未经确认的得分或胜负。回放中心沿用已被视觉模型评审的动作峰值关键帧；短回合会缩短回放源窗口，所有预告与回放都限定在该回合剪辑范围内。

## 准备与运行

编排器需要 Python 3.10+、NumPy、PATH 中的 FFmpeg/FFprobe，当前使用 Linux 文件锁。安装编排器依赖：

```bash
pip install -r tools/volleyball-top-plays/requirements.txt
python tools/volleyball-top-plays/run_match.py --video data/样例视频/1.mp4 --top-k 5
```

本地依赖：

| 项目 | 默认解释器 | 必要资产 |
| --- | --- | --- |
| `tools/volleyball_analytics` | `.venv-inference/bin/python` | 本机 `run_sample.py`、`run_batch_video.py` 适配入口；VideoMAE、球、动作、球场和人体权重；PyAV、PyTorch |
| `tools/fast-volleyball-tracking-inference` | `.venv/bin/python` | `VballNetV1_seq9_grayscale_330_h288_w512.onnx`；NumPy、OpenCV、Pillow、SciPy、ONNX Runtime |
| `tools/volleyball-highlights` | `.venv/bin/python` | YOLOv8n、EasyOCR 权重；仅指定号码时推理 |

通过 `--analytics-python`、`--tracking-python`、`--player-python` 可指定独立环境。中文字体默认使用本机 Noto Sans CJK；其他机器以 `--font /path/to/chinese-font.ttf` 指定。跟球渲染复用上游 `make_reels.py` 的裁剪和平滑函数；由本项目负责基于 PTS 解码、全场小画面、标题与原声封装，解决上游直接导出缺少音轨和名义帧率时间漂移的问题。

```bash
# 十佳与目标球员；号码识别没有两个高置信度命中时不提升权重
python tools/volleyball-top-plays/run_match.py --video match.mp4 --top-k 10 --focus-player 12

# 完全本地的规则版
python tools/volleyball-top-plays/run_match.py --video match.mp4 --ranker rules

# 禁用历史缓存发现，让三个组件实际推理（号码组件仅在指定号码时运行）
python tools/volleyball-top-plays/run_match.py --video match.mp4 --no-cache-discovery --device cpu

# 调整排序配置后从相应步骤继续；不必重复全片检测
python tools/volleyball-top-plays/run_match.py --video match.mp4 --rerun-from rank
```

程序默认查找明确记录同一素材路径、完整帧数的历史分析与追踪结果，并逐帧校验轨迹与比赛分析时间轴。可用 `--analytics-cache` 和 `--tracking-cache` 显式指定。缓存导入会复制必要原始证据，记录原路径、SHA-256、项目提交、模型哈希和参数；不会移动或覆盖上游文件。历史缓存没有源视频内容哈希，身份依据是路径、帧数及时间戳，所以更换同名素材后应禁用缓存发现。

## 大模型

默认读取项目根目录的 `llm_api.json`，格式如下；不要把真实密钥提交到 Git：

```json
{
  "llm": {
    "provider": "openai_compatible",
    "base_url": "https://your-provider.example/v1",
    "api_key": "YOUR_KEY",
    "model": "YOUR_VISION_MODEL"
  }
}
```

可用 `--llm-config` 改路径，或使用 `VOLLEYMOLE_API_KEY` / `OPENAI_API_KEY`、`VOLLEYMOLE_API_BASE`、`VOLLEYMOLE_MODEL` 环境变量。若配置中另有 `rank` 节点，它通常是文本重排模型，不用于生成剪辑单。

接口使用 Chat Completions 的图像输入和 JSON Schema，格式依照 [图像输入文档](https://developers.openai.com/api/docs/guides/images-vision) 与 [结构化输出文档](https://developers.openai.com/api/docs/guides/structured-outputs)。主模型如果明确返回“不支持多模态”，程序会查询同一服务的模型列表，选择视觉模型分批评审，再将视觉证据交给原主模型排名。也可通过 `--vision-model`、`VOLLEYMOLE_VISION_MODEL` 或 `llm.vision_model` 显式指定视觉模型。

本机配置的 GLM-5.2 实测不接受图像；同一服务的 Kimi-K3 和 Qwen3.8-27B 已通过真实排球截图测试。默认自动选择 Qwen3.8-27B 评审截图，再由 GLM-5.2 生成最终剪辑单，记录模式为 `vision_then_text_api`。如果主模型自身支持图像，则直接完成多模态排序。

默认预筛 25 个候选，每回合三张 640 像素宽截图，以及回合统计；不发送整场视频。视觉评审每次处理一个候选，避免大批量图像请求超时，结果和请求 token 用量保存在 `semantic/batch_*.json`；输入图片、统计、模型和提示词都匹配时才复用。请求失败、拒绝、未结束、未知 ID、数量不对、重复、越界或裁断回合，均会降级为规则排序。

`ranking_log.json` 记录是否实际使用大模型及失败类别，`edit_decision.json` 不会把规则输出伪装成语义评审。修复接口后可用 `--rerun-from rank` 重新请求。

## 算法与边界

视频显示时间戳是唯一时间基准；不能用原片帧号直接除以 30。本片采用约 0.5 秒运动窗口，结合球轨迹连续性、位于球员上身以上的空中运动、比赛状态和动作证据。短暂的 `NO_PLAY` 跳变可由连续空中运动连接；静止的持球预测不因 `PLAY` 标签而成为精彩回合。

轨迹候选与单独的比赛状态候选都保留，短片段、无有效轨迹和疑似停顿明确标记淘汰。评分综合长度、轨迹覆盖、转折、合并后的动作事件、参与人数和可靠号码出现比例。所有参数见 `defaults.json`，可用 `--config overrides.json` 覆盖。

这是启发式首版，候选清单不等于人工真值。远景号码、球出画、低球、多人遮挡、热身与休息会影响召回和边界；状态与动作模型也有机位域偏差。规则标题不声称识别到得分或胜负。视频级正式召回率需要另建独立人工标注集。

## 运行目录与恢复

`runs/<素材名>-topK/` 包含：

- `analytics/`：原始检测、汇总、运动窗口及来源登记。
- `tracking/`：逐帧球坐标、源 PTS、每回合轨迹。
- `player/`：指定号码的检测时间索引，或未启用标记。
- `previews/`：每个候选开局、动作峰值、结尾截图。
- `semantic/`：视觉模型分批评审、截图来源指纹与 API 用量。
- `match_manifest.json`：完整候选、得分明细、边界、证据索引。
- `edit_decision.json`：排名、标题、解释、安全剪辑点、所有未入选原因。
- `clips/`、`top5.mp4` 或 `top10.mp4`：逐回合视频与合集。
- `state.json`、日志、`render_report.json`、`verification.json`：阶段状态、错误、渲染和完整解码校验。
- `clips_lively/`、`extras_lively/`、`top5_lively.mp4` 或 `top10_lively.mp4`：活力版完整回合、预告/回放/转场与合集，不覆盖原版。
- `render_report_lively.json`：逐段源时间、播放速率、成片位置和类型；`verification_lively.json`、`alignment_verification_lively.json` 保存完整解码及画面/声音对齐检查。
- `timing_lively.json`：最近一次实际制作耗时；`timing_latest.json`：最近一次命令耗时；`timings/` 保留每次记录。

每阶段完成后原子写入状态，哈希匹配才复用。中断或失败后重复同一命令即可；成功步骤保留，失败步骤重试。少于 K 个有效回合会明确失败，不复制片段凑数量。`--stop-after manifest` 或 `--stop-after rank` 可检查中间结果。

计时采用墙钟时间，从命令启动至成片验证结束，包含源文件检查、哈希校验、实际执行阶段和最终验证。报告将当前缓存检查时间与历史推理时间分开；复用缓存的渲染时间不能当作首次全流程推理耗时。`--rerun-from render --style lively` 可单独重做并计时当前样式，不重复分析和排名。

## 验证

```bash
python -m unittest discover -s tools/volleyball-top-plays/tests -v
```

覆盖两种数量的决策约束、回合分割、源时间错位、缺帧、持球排除、号码低置信度处理、API 无效返回和错误降级、缺失产物与中断恢复，以及快切/回放时间范围、完整回合保留和本次计时。每次正式成片后自动完整解码合集及全部片段，检查分辨率、音视频起止、时长和排名数，并对照源画面及原声波形。

实际编码的故障注入测试（生成短小的合成十佳，不调用外部 API）：

```bash
python tools/volleyball-top-plays/tests/render_fixture.py \
  --output runs/verification/top10-api-failure \
  --python tools/fast-volleyball-tracking-inference/.venv/bin/python \
  --api-failure
```

它仅模拟 API 传输断网，其余降级决策、静音音轨、缺少球轨迹时的居中构图、十段合并和完整解码都实际运行。

测试新版效果时添加 `--style lively`，并将输出目录改为 `runs/verification/lively-top10`。

正式成片也可单独重复对照源画面和原声波形，活力版结果保存为 `alignment_verification_lively.json`：

```bash
tools/fast-volleyball-tracking-inference/.venv/bin/python \
  tools/volleyball-top-plays/check_alignment.py --run runs/1-top5 --style lively
```

第三方模型与源码按各自许可证独立安装，本仓库不重新分发权重。尤其 `volleyball_analytics` 的 GPL 许可证需在部署和分发时按上游原文处理；本编排器仅调用本地检出。
