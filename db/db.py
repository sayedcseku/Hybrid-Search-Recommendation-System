from typing import Optional, Iterable, Dict, Any
import os
from sqlalchemy import create_engine, Table, Column, MetaData, String, Text
from sqlalchemy.dialects.postgresql import JSONB, insert as pg_insert
from sqlalchemy.exc import ProgrammingError
from typing import Optional, Iterable, Dict, Any, List, Tuple
import os
from sqlalchemy import create_engine, Table, Column, MetaData, String, Text, select
from sqlalchemy.dialects.postgresql import JSONB, insert as pg_insert
from sqlalchemy.exc import ProgrammingError, DatabaseError
try:
	from pgvector.sqlalchemy import Vector
except Exception:
	Vector = None  # pgvector not installed in this environment

import numpy as np


def get_engine_from_env() -> create_engine:
	"""Reads DATABASE_URL from env or uses a default local Postgres.

	Expected format: postgresql://user:pass@host:port/dbname
	"""
	url = os.environ.get("DATABASE_URL")
	if not url:
		url = "postgresql://postgres:postgres@localhost:5432/yelp_recsys"
	return create_engine(url)


def create_business_table(engine=None, vector_dim: int = 512):
	engine = engine or get_engine_from_env()
	meta = MetaData()

	# Ensure pgvector extension exists if possible
	with engine.connect() as conn:
		try:
			conn.execute("create extension if not exists vector")
		except Exception:
			# some environments can't create extensions; continue and fall back to JSONB for vectors
			pass

	# Use Vector type when available; otherwise store vectors as JSONB/list
	img_col_type = Vector(vector_dim) if Vector is not None else JSONB

	business = Table(
		"businesses",
		meta,
		Column("business_id", String, primary_key=True),
		Column("name", Text),
		Column("raw", JSONB),
		Column("image_vector", img_col_type),
	)

	try:
		meta.create_all(engine)
	except ProgrammingError:
		# Retry using JSONB for the vector column
		meta = MetaData()
		business = Table(
			"businesses",
			meta,
			Column("business_id", String, primary_key=True),
			Column("name", Text),
			Column("raw", JSONB),
			Column("image_vector", JSONB),
		)
		meta.create_all(engine)

	return business


def upsert_business(engine, business_row: Dict[str, Any], vector: Optional[Iterable[float]] = None):
	"""Insert or update a business record with optional image vector."""
	conn = engine.connect()
	meta = MetaData()
	meta.reflect(bind=engine)
	table = meta.tables.get("businesses")
	if table is None:
		raise RuntimeError("businesses table does not exist; call create_business_table first")

	values: Dict[str, Any] = {
		"business_id": str(business_row.get("business_id")),
		"name": business_row.get("name"),
		"raw": business_row,
	}
	if vector is not None:
		# store as plain Python list — works for both pgvector and JSONB fallback
		values["image_vector"] = list(vector)

	# Upsert using PostgreSQL ON CONFLICT
	stmt = pg_insert(table).values(**values)
	do_update_stmt = stmt.on_conflict_do_update(
		index_elements=[table.c.business_id],
		set_=values,
	)
	try:
		conn.execute(do_update_stmt)
	finally:
		conn.close()


def search_businesses(engine, query_vector, limit: int = 10) -> List[Tuple[str, str, Any, float]]:
	"""Run a nearest-neighbor search using pgvector <-> operator, falling back to Python L2 if needed.

	Returns list of tuples: (business_id, name, raw, distance)
	"""
	conn = engine.connect()
	try:
		# Try fast path using pgvector operator
		try:
			sql = "SELECT business_id, name, raw, image_vector <-> %s as distance FROM businesses ORDER BY distance LIMIT %s"
			res = conn.execute(sql, (list(query_vector), limit)).fetchall()
			return res
		except Exception:
			# Fall back to Python-side nearest neighbor if SQL/vector operator not available
			pass

		# Fallback: load vectors and compute distances in Python
		meta = MetaData()
		table = Table("businesses", meta, autoload_with=engine)
		sel = select(table.c.business_id, table.c.name, table.c.raw, table.c.image_vector)
		rows = conn.execute(sel).fetchall()
		items: List[Tuple[str, str, Any, float]] = []
		q = np.array(query_vector, dtype=float)
		for r in rows:
			bid, name, raw, vec = r[0], r[1], r[2], r[3]
			if vec is None:
				continue
			try:
				v = np.array(vec, dtype=float)
			except Exception:
				continue
			dist = float(np.linalg.norm(q - v))
			items.append((bid, name, raw, dist))
		items.sort(key=lambda x: x[3])
		return items[:limit]
	finally:
		conn.close()

