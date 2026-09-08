<p align="center">
  <img src="docs/brand/volleymole-logo.svg" alt="VolleyMole" width="1000">
</p>

<h1 align="center">🏐 VolleyMole</h1>

<p align="center">
  一款本地优先的桌面 Web 应用，从排球比赛长视频中“挖”出一个个回合。
</p>

<p align="center">
  <a href="README.md">English</a> · <strong>简体中文</strong>
</p>

## ✨ 项目功能

VolleyMole 将符合条件的连续比赛视频处理成可复核、可独立导出的回合片段。Web 界面和视频处理服务运行在同一台电脑上，逐帧检测默认使用本机 GPU。

第一版工作流刻意保持精简：

1. 📂 选择本地比赛视频。
2. 🔍 在本机运行分析。
3. ✂️ 复核检测到的回合边界。
4. 📦 将确认后的回合分别导出。

## 🎯 第一版范围

VolleyMole v1 主要面向：

- 单块排球场清晰可见；
- 固定或轻微抖动机位；
- 没有频繁切镜的连续比赛画面；
- 球场及主要飞行区域基本保持可见。

### 🚧 第一版不保证的场景

- 电视转播剪辑或频繁切镜
- 多机位或多块球场同时入镜
- 频繁变焦
- 长时间离开球场画面

球员身份识别、个人集锦、动作分类、战术统计、自动生成 9:16 竖屏视频和生成式转场均为延期方向，不属于第一版交付承诺。

## 🧩 系统架构

```text
React + TypeScript + Vite 界面
               │ localhost HTTP
               ▼
Python 本地服务
               │
               ├── 视频元数据与任务编排
               └── 本机 GPU 推理
```

视频保留在用户电脑中。当前骨架不包含云端后端，也没有视频上传链路。

## 🚀 快速开始

### 环境要求

- Node.js 20+
- Python 3.11+
- 较新版本的 [uv](https://docs.astral.sh/uv/)

### 启动 Web 界面

```bash
npm install
npm run dev
```

### 启动本地处理服务

打开另一个终端：

```bash
cd services/inference
uv sync
uv run uvicorn volleymole_service.main:app --reload --port 8000
```

访问 [http://localhost:5173](http://localhost:5173)。服务健康检查地址为 [http://localhost:8000/api/health](http://localhost:8000/api/health)。

## 📁 仓库结构

```text
src/                         桌面 Web 界面
services/inference/          Python 本地处理服务
services/inference/tests/    服务接口测试
```

## 🛠️ 开发状态

VolleyMole 当前处于第一版早期骨架阶段。下一步实施重点是接入本地模型，以及完成可编辑的回合复核工作流。

### 独立阶段 0/1 引擎

独立分析引擎位于 `src/volleycut`，附带根目录 Python 依赖、测试和原图审阅工具源码，尚未接入现有界面或 HTTP 服务。

独立 Python 3.12 环境与 CLI 见[引擎指南](README_ENGINE.md)，数据准备及复现限制见[本地数据说明](docs/阶段0-1/数据与复现说明.md)。用户素材、标注文件、数据清单、视频指纹、实验报告、模型权重和凭据均不公开上传。

验证集冻结、独立质量基线、验收配置及用户实测节时四项门禁均保持**未通过**。上传源码和说明文档不代表后续产品阶段已获放行。

## 📄 许可证

[MIT](LICENSE)
