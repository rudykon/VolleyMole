from fastapi import FastAPI

app = FastAPI(title="VolleyMole local service", version="0.1.0")


@app.get("/api/health")
def health() -> dict[str, str]:
    """Report whether the local process is reachable."""
    return {"status": "ready", "processing": "local"}
