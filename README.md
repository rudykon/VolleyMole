# VolleyMole

把单机位日常排球比赛自动剪成竖屏五佳球或十佳球：回合分割 → 精彩评分 → 排名成片。

第一版编排器位于 [tools/volleyball-top-plays](tools/volleyball-top-plays/README.md)。复用本地三个上游项目，不训练模型，各上游使用独立 Python 环境。

```bash
python tools/volleyball-top-plays/run_match.py \
  --video data/样例视频/1.mp4 --top-k 5
```

输出 `runs/1-top5/top5.mp4`、五个独立片段、完整回合清单、剪辑决策及验证报告。重新运行同一命令会检查输入和产物哈希，并复用已完成步骤。

读取本地 `llm_api.json` 的 `llm` 配置，或使用 `VOLLEYMOLE_API_KEY`、`VOLLEYMOLE_API_BASE`、`VOLLEYMOLE_MODEL` 环境变量。只发送预筛候选的结构化统计及每回合最多三张缩小截图；原始整场视频不上传。API 不可用或决策不合法时保留错误分类并按规则导出。`--ranker rules` 完全关闭 API 请求。

模型、素材、运行结果、API 配置和上游完整检出均不进入 Git。项目范围见 [核心功能需求](docs/核心功能需求.md)。
