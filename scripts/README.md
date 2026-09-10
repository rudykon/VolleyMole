# 脚本说明

从仓库根目录执行；除 `check_release.py` 外，通常使用项目 `.venv/bin/python`。

| 脚本 | 用途与前提 |
| --- | --- |
| `check_release.py` | 仅使用 Python 标准库；检查待发布文件，支持 `--history` |
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
