# 参与开发

## 项目结构

```text
src/volleymole/   CLI、模型适配、推理、回合、排名与渲染
  assets/        品牌、插画、字体及来源记录
  licenses/      改编源码的版权与许可说明
tests/           自动测试及另行执行的真实模型测试
scripts/         发布检查、性能对照和历史验收工具
docs/            使用说明、性能记录与历史方案
```

`models/`、`data/`、`runs/` 和 `.venv/` 是本地目录，不提交 Git。项目运行不依赖 `tools/`。

## 开发与验证

```bash
uv sync --locked
.venv/bin/volleymole assets fetch
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
python3 scripts/check_release.py
git diff --check
```

自动测试不需要下载模型、GPU 或真实 API。完整本地测试还会检查通过 Releases 安装、或开发目录已有的生成式设计素材；GitHub Actions 中的 [Engineering checks](.github/workflows/engineering-checks.yml) 只运行可发布的核心与双榜回归，并用 4 秒合成音视频确认新榜单入口会生成原画面／原声对齐报告。该 Actions 结果不代表真实模型精度、GPU 兼容性、外部 API 能力或真实比赛成片已验收。

真实推理和性能脚本需要本地模型与素材，运行方法见 [脚本说明](scripts/README.md)。修改推理时，应核对帧覆盖、PTS、模型窗口、原始证据与回合边界；修改呈现时，应核对成片解码、帧数、音视频起点／时长、原片画面／原声内容对齐与标题检查。

## 提交内容

- 描述改动目的、复现步骤和已执行的验证；性能数字注明硬件、缓存状态、输入规模及计时范围。
- 保留模型精度、输入尺寸和时序约束；如改变结果，应明确给出差异与验证依据。
- 不提交真实密钥、用户录像、模型权重、运行日志或虚拟环境。配置示例使用占位符。
- 新增第三方代码或素材时同步更新来源及许可记录；不改写已有第三方版权声明。

## GitHub 发布准备

发布检查脚本只扫描，不会自动暂存、提交或推送。它检查当前待发布文件；可用 `--history` 扫描历史对象中的常见凭据格式及大文件。启发式检查不能保证发现所有敏感内容，提交前仍需阅读 `git diff --cached`。

项目自有代码采用 [MIT License](LICENSE)，自有代码贡献应使用相同许可。新增第三方内容仍需保留其原始许可与来源记录，不能将依赖或模型权重统一改标为 MIT。模型与录像不作为源码仓库内容发布。
