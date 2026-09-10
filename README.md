# VolleyMole

把单机位日常排球比赛自动剪成竖屏五佳球或十佳球：回合分割 → 精彩评分 → 排名成片。

统一实现位于 [src/volleymole](src/volleymole)。状态／动作识别、VballNet、独立号码 OCR、回合分割、排名与剪辑在一个可安装的 Python 包中运行，不需要手工检出三个上游项目，也不训练模型。安装、权重和许可说明见 [统一包安装与运行](docs/统一包安装与运行.md) 与 [第三方来源](THIRD_PARTY.md)。

```bash
uv sync --locked
.venv/bin/volleymole models --directory models fetch
.venv/bin/volleymole run --video data/样例视频/1.mp4 --top-k 5 --device cuda:0

# 不调用大模型的十佳；同源、同模型配置复用全场分析
.venv/bin/volleymole run --video data/样例视频/1.mp4 --top-k 10 --device cuda:0 --ranker rules
```

默认输出插画活力版 `runs/1-top5/top5_lively.mp4`：Codex 生图插画、手写风格标题、5 秒精彩快切片头、完整五佳回合、短慢回放和 3 秒插画标题转场。回合与回放顶部、转场显示生图横幅承托的“五佳球 · 第几球／十佳球 · 第几球”，转场顶部展示用户提供的 `volleymole.svg` 标志，标题完整停留 2.2 秒，不使用“#1”式标号。保留现场原声，回放同步降速。原版 `top5.mp4` 不被覆盖，前两次活力样式另保留为 `top5_lively_v1.mp4`、`top5_lively_v2.mp4`；使用 `--style classic` 可生成原版样式。

完整回合清单、剪辑决策、源时间线、验证报告和本次耗时都保存在运行目录。`timing_lively.json` 记录最近一次实际制作的总耗时及分阶段耗时，明确标记复用缓存；重复运行只校验缓存时，不覆盖这份制作时间记录。

正常回合和回放的顶部名次标题采用透明叠加，不再使用整条黑色底栏；比赛画面延伸到顶部，文字以细描边保证对比度。回放提示也保持透明。

读取本地 `llm_api.json` 的 `llm` 配置，或使用 `VOLLEYMOLE_API_KEY`、`VOLLEYMOLE_API_BASE`、`VOLLEYMOLE_MODEL` 环境变量。只发送预筛候选的结构化统计及每回合最多三张缩小截图；原始整场视频不上传。API 不可用或决策不合法时保留错误分类并按规则导出。`--ranker rules` 完全关闭 API 请求。

模型、素材、运行结果、API 配置和上游完整检出均不进入 Git。项目范围见 [核心功能需求](docs/核心功能需求.md)。

分析默认共用一次 PTS 解码和人体检测；辅助球检测仅在已有球证据缺失时运行。低球与短遮挡保留原始预测、辅助检测来源和边界不确定性，不把插值当检测、不声称自动确认得分。`--device cuda:0` 是单卡推理；显式使用 `--devices cuda:0,cuda:1,cuda:2,cuda:3` 可将状态、动作、人物、球轨迹分到四卡并行。两种模式都只解码一次，渲染仍使用 CPU；四卡分析不等于四倍成片速度。

本机同源整场四卡分析加清单实测 10 分 16 秒，对照历史单卡 17 分 28 秒，观察到 1.70 倍加速；57,531 帧原始证据逐字节一致。计时范围、非独占负载限制和复现方法见 [四卡 GPU 实测](docs/四卡GPU实测.md)。

完整阶段验证与基线对照见 [整合实施与验收台账](docs/单仓库整合实施与验收.md)。旧 `tools/volleyball-top-plays/run_match.py` 仅作为转发入口保留，维护新代码请修改 `src/volleymole`。上游检出和旧成片保留供追溯，不参与新包运行。
