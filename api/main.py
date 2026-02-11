from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
import uvicorn
import os
from db.db import get_engine_from_env, create_business_table, upsert_business, search_businesses
from sentence_transformers import SentenceTransformer
import json
from tempfile import NamedTemporaryFile

DEFAULT_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

app = FastAPI(title="Yelp Recsys API")


@app.on_event("startup")
def startup_event():
    # Initialize and store shared resources
    engine = get_engine_from_env()
    app.state.engine = engine
    # Load embedding model once
    model_name = os.environ.get("EMBEDDING_MODEL", DEFAULT_MODEL)
    app.state.model_name = model_name
    app.state.model = SentenceTransformer(model_name)


@app.on_event("shutdown")
def shutdown_event():
    engine = getattr(app.state, "engine", None)
    if engine is not None:
        try:
            engine.dispose()
        except Exception:
            pass


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/setup")
def setup(vector_dim: int = 384):
    engine = getattr(app.state, "engine", None) or get_engine_from_env()
    create_business_table(engine, vector_dim)
    return {"created": True}


def _process_file(path: str, model_name: str | None = None):
    # Use shared model if model_name matches loaded one; else load a new model
    if model_name is None or model_name == getattr(app.state, "model_name", None):
        model = app.state.model
    else:
        model = SentenceTransformer(model_name)

    engine = getattr(app.state, "engine", None) or get_engine_from_env()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            name = r.get("name", "")
            text = name + " " + (r.get("categories") or "")
            vec = model.encode(text, convert_to_numpy=True)
            upsert_business(engine, r, vec.tolist())


@app.post("/ingest")
async def ingest(file: UploadFile = File(...), background: BackgroundTasks = None, model: str = None):
    # Save upload to temp file and process in background
    tmp = NamedTemporaryFile(delete=False, suffix=".jsonl")
    content = await file.read()
    tmp.write(content)
    tmp.flush()
    tmp.close()

    chosen_model = model or getattr(app.state, "model_name", DEFAULT_MODEL)
    if background is not None:
        background.add_task(_process_file, tmp.name, chosen_model)
        return JSONResponse({"started": True, "path": tmp.name})
    else:
        _process_file(tmp.name, chosen_model)
        return {"processed": True}


@app.get("/count")
def count():
    engine = getattr(app.state, "engine", None) or get_engine_from_env()
    conn = engine.connect()
    res = conn.execute("select count(*) from businesses").scalar()
    conn.close()
    return {"count": int(res)}


@app.get("/search")
def search(q: str, k: int = 10, model: str | None = None):
    """Compute embedding for query text and return nearest businesses."""
    # Use loaded model when possible
    model_obj = app.state.model if (model is None or model == getattr(app.state, "model_name", None)) else SentenceTransformer(model)
    vec = model_obj.encode(q, convert_to_numpy=True)
    engine = getattr(app.state, "engine", None) or get_engine_from_env()
    rows = search_businesses(engine, vec, limit=k)
    out = []
    for r in rows:
        # r: business_id, name, raw, distance
        out.append({"business_id": r[0], "name": r[1], "raw": r[2], "distance": float(r[3])})
    return {"results": out}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
