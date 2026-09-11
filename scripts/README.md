# 脚本说明

从仓库根目录执行；除 `check_release.py` 外，通常使用项目 `.venv/bin/python`。

| 脚本 | 用途与前提 |
| --- | --- |
| `evaluate_events.py` | 按比赛分区评估候选覆盖率、前五质量、解释与剪辑完整性，以及同条件端到端中位数/P95；需要人工标注和运行记录 |
| `check_release.py` | 仅使用 Python 标准库；检查待发布文件，支持 `--history` |
| `preview_art_themes.py` | 无需 GPU、模型或 API，检查透明素材并导出实际标题卡、角标及 HTML 图集 |
| `preview_title_templates.py` | 十套中英标题与五套插画组合预览；无需 GPU、模型或 API |
| `preview_transitions.py` | 五套动画转场视频预览、编码校验与透明 ProRes 4444 导出；需要 FFmpeg，无需 GPU 或 API |
| `preview_design_suites.py` | 五套完整设计的动画、角标与图集；`--language en/zh/both` 选择语言，默认 1080p；可复用已有运行制作短实拍样片 |
| `prepare_title_fonts.py` | 开发用字体集合 SC 字面提取，需要 FontTools；正常运行无需执行 |
| `benchmark_optimizations.py` | 顺序比较四种推理配置；需要四卡 CUDA、模型与录像 |
| `benchmark_render.py` | 同一剪辑单的串行/并行渲染对照；需要完整参考运行目录 |
| `compare_optimized_run.py` | 比较两次运行的检测、回合、预览和成片 |
| `compare_gpu_runs.py` | 四卡与已有分析结果对照；需要本地运行产物 |
| `compare_integration_runs.py` | 整合前后结果对照；需要本地基线 |
| `verify_installed_package.py` | 在隔离的已安装 wheel 环境验证真实推理；需要模型与素材 |
| `verify_real_resume.py` | 检查真实运行的断点恢复；需要已有运行目录 |
| `report_dependency_licenses.py` | 汇总当前安装依赖的许可元数据 |

`capture_integration_baseline.py` 与 `final_integration_audit.py` 是历史整合审计脚本，依赖已退役的 `tools/` 检出、旧兼容测试和指定本机归档。保留供追溯，不属于新克隆仓库的安装、测试或发布步骤。

性能复现命令及结果见 [性能优化与验证](../docs/性能优化与验证.md)。所有新的基准实验应使用独立输出目录。
