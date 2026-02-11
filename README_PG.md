This project adds PostgreSQL + pgvector integration and a FastAPI ingestion/search API.

Quick start (Windows PowerShell):

1. Start Postgres with pgvector extension available. Using docker-compose:

```powershell
docker compose up -d
```

2. Install requirements (prefer a venv):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

3. Create the pgvector extension and table (the app will create tables, but extension must be created once):

```powershell
# connect to DB (example using psql)
psql -h localhost -U postgres -d yelp_vectors -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

4. Run the API:

```powershell
$env:DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/yelp_vectors"; uvicorn api.main:app --reload
```

5. Ingest NDJSON via `POST /ingest/json` or use `scripts/ingest_to_pg.py --json path/to/file`.

Notes:
- Default embedding model: `all-MiniLM-L6-v2`. Change with `EMBEDDING_MODEL` env var.
- Adjust vector dim in `db/db.py` if you use a different model with another dimension.
