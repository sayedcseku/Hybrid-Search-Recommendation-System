"""
Ingest Yelp JSON lines file into Postgres with pgvector vectors.

Usage (example):
  python scripts/ingest_to_pg.py --json_file Data/yelp/yelp_academic_dataset_business.json --batch 128

This script expects a Postgres server with pgvector extension enabled and a DATABASE_URL env var.
"""
import argparse
import json
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from db.db import get_engine_from_env, create_business_table, upsert_business


def iter_jsonlines(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json_file", required=True)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    model = SentenceTransformer(args.model)
    engine = get_engine_from_env()
    create_business_table(engine)

    path = Path(args.json_file)
    rows = iter_jsonlines(path)

    batch = []
    for r in tqdm(rows):
        business_id = r.get("business_id")
        name = r.get("name", "")
        text = name + " " + (r.get("categories") or "")
        batch.append((r, text))
        if len(batch) >= args.batch:
            texts = [t for (_, t) in batch]
            vecs = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            for (row, _), v in zip(batch, vecs):
                upsert_business(engine, row, v.tolist())
            batch = []

    if batch:
        texts = [t for (_, t) in batch]
        vecs = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        for (row, _), v in zip(batch, vecs):
            upsert_business(engine, row, v.tolist())

    print("Ingestion complete.")


if __name__ == "__main__":
    main()
