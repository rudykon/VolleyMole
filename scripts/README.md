# 脚本说明

从仓库根目录执行；除 `check_release.py` 外，通常使用项目 `.venv/bin/python`。

| 脚本 | 用途与前提 |
| --- | --- |
| `install_sound_model.py` | 下载并校验 PANNs 作者权重、导出本地逐帧声音模型；不训练或新增标签 |
| `prepare_public_audio.py` / `evaluate_sound_events.py` | 采集完整 ESC-50 原音频与标签，自动评测笑声和掌声片段分类 |
| `prepare_sound_calibration.py` | 用 FSD50K 原 dev/train 标签固定抽样，排除 ESC-50 源 ID；供独立阈值校准 |
| `calibrate_sound_thresholds.py` | 仅用固定开发集原标签选择候选发现阈值，输出与模型哈希绑定的侧文件；可复用 ESC-50 预测作非盲后续检查 |
| `verify_sound_model_reference.py` | 在隔离的参考依赖环境与作者原实现做数值一致性比较 |
| `acquire_volleyball_benchmark.py` | 采集 VNL-STES 原标签与连续帧，生成原 train/val/test 隔离目录和封存清单 |
| `evaluate_volleyball_actions.py` | 使用配置的视觉 API 对原测试动作做时间匹配评测，答案不进入请求；报告不完整请求 |
| `compare_action_understanding.py` | 在原 val 比较视觉理解方法；test 强制冻结配置和代码指纹，失败请求保留漏检 |
| `train_action_spotter.py` | 原 train/val 训练冻结 ResNet18＋时序 GRU，并独立评测封存 test；使用现有本地 GPU |
| `evaluate_existing_action_detector.py` | 用同一原标签时间匹配评测现有 YOLO，保留其上游训练来源未知限制 |
| `export_action_evidence.py` | 流式导出与源 SHA 和真实 PTS 绑定的 25 FPS 动作候选；预算截止输出 partial |
| `blooper_benchmark.py` | 生成匿名原速素材、空表与离线 HTML，导入独立真人评分并计算前五质量和一致性；不自动标注 |
| `prepare_blooper_review.py` / `build_blooper_review_report.py` | 为已有声音候选生成原声上下文与PTS时间帧，打包已有模型复核记录和可播放报告 |
| `review_local_blooper_api.py` / `transcribe_blooper_context.py` | 对已授权上传的本地候选进行有限视觉复核与音频转写；声音关联保留未知，转写不是笑声真值 |
| `prepare_svhighlights.py` / `evaluate_svhighlights.py` | 采集 40 场排球原集锦对齐标签，评测严格时间对齐的预测；提供原音量基线 |
| `evaluate_events.py` | 在有对应既有人工评分及运行记录时，按比赛评估候选、前五、解释、剪辑及耗时；动作标签不能替代主观评分 |
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
