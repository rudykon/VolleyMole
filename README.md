# VolleyMole

VolleyMole is a local-first desktop web application for turning suitable volleyball match videos into reviewable rally clips. The web interface and video-processing service run on the same computer; frame-by-frame detection uses the local GPU by default.

## Version 1 scope

Version 1 targets continuous footage of one visible court from a fixed or slightly shaky camera. The court and the main ball-flight area should remain substantially visible.

The initial workflow is deliberately small:

1. select a local match video;
2. run local analysis;
3. review detected rally boundaries;
4. export each accepted rally as an independent clip.

TV edits, multiple cameras, frequent zooms or cuts, long periods away from the court, and multiple courts in one frame are outside the guaranteed v1 operating conditions. Player identification, personal highlight reels, action classification, tactical statistics, automatic vertical reframing, and generative transitions are deferred rather than implied by this repository.

## Architecture

```text
React + TypeScript + Vite UI
            │ localhost HTTP
            ▼
Python local service
            │
            ├── video metadata and job orchestration
            └── local GPU inference (model integration follows)
```

Uploaded videos are represented by local file metadata in the current scaffold. No cloud backend or upload path is included.

## Quick start

Requirements: Node.js 20+, Python 3.11+, and a recent `uv` installation.

```bash
npm install
npm run dev
```

In another terminal:

```bash
cd services/inference
uv sync
uv run uvicorn volleymole_service.main:app --reload --port 8000
```

Open `http://localhost:5173`. The service health endpoint is `http://localhost:8000/api/health`.

## Repository layout

```text
src/                         desktop web UI
services/inference/          local Python processing service
services/inference/tests/    service contract tests
```

## Status

Early v1 scaffold. Model integration and the editable rally-review workflow are the next implementation milestones.

## License

MIT
