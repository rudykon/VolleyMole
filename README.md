# VolleyMole

把单机位日常排球比赛自动剪成竖屏五佳球或十佳球：回合分割 → 精彩评分 → 排名成片。

第一版编排器位于 [tools/volleyball-top-plays](tools/volleyball-top-plays/README.md)。复用本地三个上游项目，不训练模型，各上游使用独立 Python 环境。

```bash
python tools/volleyball-top-plays/run_match.py \
  --video data/样例视频/1.mp4 --top-k 5
```

默认输出活力版 `runs/1-top5/top5_lively.mp4`：彩色活泼标题、4 秒精彩快切片头、完整五佳回合、短慢回放和连接动画。保留现场原声，回放同步降速。原版 `top5.mp4` 不被覆盖，使用 `--style classic` 可生成原版样式。

完整回合清单、剪辑决策、源时间线、验证报告和本次耗时都保存在运行目录。`timing_lively.json` 记录最近一次实际制作的总耗时及分阶段耗时，明确标记复用缓存；重复运行只校验缓存时，不覆盖这份制作时间记录。

读取本地 `llm_api.json` 的 `llm` 配置，或使用 `VOLLEYMOLE_API_KEY`、`VOLLEYMOLE_API_BASE`、`VOLLEYMOLE_MODEL` 环境变量。只发送预筛候选的结构化统计及每回合最多三张缩小截图；原始整场视频不上传。API 不可用或决策不合法时保留错误分类并按规则导出。`--ranker rules` 完全关闭 API 请求。

模型、素材、运行结果、API 配置和上游完整检出均不进入 Git。项目范围见 [核心功能需求](docs/核心功能需求.md)。
