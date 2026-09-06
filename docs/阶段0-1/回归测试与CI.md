# 阶段 0/1 回归测试与 CI

CI 只证明工程约束的回归检查通过，不是模型质量验收或网页闭环放行。
验证集冻结、独立质量基线、版本化验收配置、用户实测节时四项仍未通过。

## 两类检查

| 检查 | 命令 | 需要的材料 |
| --- | --- | --- |
| 全部公开工程回归 | `PYTHONPATH=src uv run --locked pytest -q -m 'not local_data'` | Python 3.12、锁定依赖、FFmpeg/FFprobe |
| 本地真实材料检查 | `PYTHONPATH=src uv run --locked pytest -q -m local_data` | 上述环境及原本的私有媒体、帧图、标注和扩展审阅记录 |
| 本地完整回归 | `PYTHONPATH=src uv run --locked pytest -q` | 两类检查的全部前置条件 |

`local_data` 不设置默认排除，也不因文件缺失调用 `skip`。显式运行真实材料检查而缺失文件时
必须失败。标注材料尚未冻结时的拒绝检查，也不能替代完整源视频/帧图片哈希验证。

公开测试运行真实 FFmpeg 生成和探测临时色块视频，检测器构造前即停止；回退策略测试拦截
ORT 会话创建，只测试控制流。校验器夹具使用人工构造的二进制源和 PNG，并测试临时目录的
冻结写入。这些都不是比赛数据、模型预测或实际真值，不能作为质量指标的分母或验收证据。

## GitHub Actions

工作流：`.github/workflows/stage01-engine.yml`。触发范围为 `main` 推送、PR 和手动运行，
使用 Ubuntu 24.04、Python 3.12、固定 uv 版本及仓库 `uv.lock`。不配置私有素材、仓库令牌
secret、GPU 或自托管运行器，不上传测试夹具、媒体或本地实验报告。

Action 固定为官方仓库发布提交的完整 SHA，权限仅 `contents: read`，checkout 不保留 Git
凭据；同一分支的新运行会取消旧运行。依据：[GitHub Actions 安全使用说明](https://docs.github.com/en/actions/reference/security/secure-use)。

CI 保持原有 `onnxruntime-gpu` 版本，但省略大型 NVIDIA 运行库，以运行不加载权重的工程检查：

```bash
uv sync --locked --python 3.12 \
  --no-install-package nvidia-cublas-cu12 \
  --no-install-package nvidia-cudnn-cu12 \
  --no-install-package nvidia-cuda-nvrtc-cu12
PYTHONPATH=src .venv/bin/python -m pytest -q -m 'not local_data'
```

此命令仅用于**独立的 CI 环境**，不要对正在使用的 GPU 环境执行（同步会移除指定库）。
它遵循 [uv 的锁文件和部分安装规则](https://docs.astral.sh/uv/concepts/projects/sync/)，
不能用于推理。实际 NVIDIA 验证须完整 `uv sync --locked`，并核查真实执行 profile 和源码指纹。

## 冻结拒绝回归

校验版本 2 显式验证回合时间、帧索引、PTS 和非空边界依据；这些门禁在 `python -O` 下仍生效。
布尔值不能冒充整数时间/帧号，空白审阅者和空白负例上下文不能冒充完成审阅。
回归同时覆盖未标帧、错误坐标、旋转后尺寸、图片损坏、源哈希改变、调参与验收重叠、
机位不足、非独立回合审阅，以及冻结记录的输入绑定与禁止覆盖。

下一阶段依然需要：完成真实帧级与独立回合级审阅、全量完整性验证、真实真值冻结、
当前模型基线、锁定验收阈值并在独立验收片段上通过。CI 绿色不能替代其中任一步。
