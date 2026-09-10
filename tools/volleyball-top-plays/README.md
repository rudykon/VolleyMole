# 历史入口（已转发到统一包）

VolleyMole 的唯一维护实现已迁至 [src/volleymole](../../src/volleymole)。
此目录的 Python 模块为兼容转发，原有 33 项测试仍在；旧素材、提示词和文档保留供追溯，
运行时实际使用包内 assets、prompts 和 defaults.json。不要在此维护第二份算法或美术配置。

```bash
uv sync --locked
.venv/bin/volleymole models --directory models verify
.venv/bin/volleymole run --video data/样例视频/1.mp4 --top-k 5 --device cuda:0
```

旧 `python tools/volleyball-top-plays/run_match.py ...` 会优先转发到根目录 `.venv`。
旧的三个独立解释器参数和自动上游缓存发现已取消；使用新命令的
`--models`、`--analysis-cache-dir`、`--evidence-cache` 或 `--no-analysis-cache`。

保留五佳/十佳、5 秒精彩快切、3 秒插画转场、中文名次、项目标志、完整回合与短慢回放、
原声和正常回合/回放的透明顶部标题。详细安装、模型来源、缓存和复现命令见
[统一包指南](../../docs/统一包安装与运行.md)，验证见
[实施台账](../../docs/单仓库整合实施与验收.md)。
第三方许可按 [THIRD_PARTY.md](../../THIRD_PARTY.md) 逐模块核对；
MIT 核心并不免除 Ultralytics 的 AGPL 条款，也不代表所有权重均获得 MIT 授权。

原版代码、环境与真实成片已冻结在本机 `runs/integration/baseline-20260910/`；
上游目录和旧成片没有删除，不是新版运行依赖。
