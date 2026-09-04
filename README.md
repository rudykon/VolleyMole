<p align="center">
  <img src="docs/brand/volleymole-logo.png" alt="VolleyMole" width="1000">
</p>

<h1 align="center">🏐 VolleyMole</h1>

<p align="center">
  A local-first desktop web app that digs rallies out of long volleyball videos.
</p>

## ✨ What it does

VolleyMole turns suitable continuous match footage into reviewable, independent rally clips. The web interface and processing service run on the same computer, and frame-by-frame detection uses the local GPU by default.

The v1 workflow is intentionally focused:

1. 📂 Select a local match video.
2. 🔍 Run local analysis.
3. ✂️ Review detected rally boundaries.
4. 📦 Export accepted rallies as individual clips.

## 🎯 Version 1 scope

VolleyMole v1 is designed for:

- one visible volleyball court;
- a fixed or slightly shaky camera;
- continuous match footage without frequent cuts;
- footage where the court and main ball-flight area remain substantially visible.

### 🚧 Outside the guaranteed v1 conditions

- TV-style edits or frequent scene changes
- Multiple cameras or courts
- Frequent zooming
- Long periods away from the court

Player identification, personal highlight reels, action classification, tactical statistics, automatic 9:16 reframing, and generative transitions are deferred features—not part of the v1 commitment.

## 🧩 Architecture

```text
React + TypeScript + Vite UI
            │ localhost HTTP
            ▼
Python local service
            │
            ├── video metadata and job orchestration
            └── local GPU inference
```

Videos stay on the user's computer. This scaffold contains no cloud backend or video-upload path.

## 🚀 Quick start

### Requirements

- Node.js 20+
- Python 3.11+
- A recent [uv](https://docs.astral.sh/uv/) installation

### Start the web interface

```bash
npm install
npm run dev
```

### Start the local processing service

In another terminal:

```bash
cd services/inference
uv sync
uv run uvicorn volleymole_service.main:app --reload --port 8000
```

Open [http://localhost:5173](http://localhost:5173). The service health endpoint is [http://localhost:8000/api/health](http://localhost:8000/api/health).

## 📁 Repository layout

```text
src/                         desktop web UI
services/inference/          local Python processing service
services/inference/tests/    service contract tests
```

## 🛠️ Development status

VolleyMole is an early v1 scaffold. The next implementation milestones are local model integration and an editable rally-review workflow.

## 📄 License

[MIT](LICENSE)
