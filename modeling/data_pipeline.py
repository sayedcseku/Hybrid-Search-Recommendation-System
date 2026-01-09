import json
import re
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd


def read_json_lines(path: Path, max_rows: Optional[int] = None) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_rows is not None and i >= max_rows:
                break
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def normalize_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def attributes_to_text(attrs) -> str:
    if not isinstance(attrs, dict):
        return ""
    parts = []
    for k, v in attrs.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                parts.append(f"{k}_{kk}:{vv}")
        else:
            parts.append(f"{k}:{v}")
    return " ".join(parts)


def load_yelp(root: str, max_rows: Optional[Dict[str, int]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load Yelp business + review JSONL tables and create text fields used by models.
    """
    root = Path(root)
    max_rows = max_rows or {}

    b = read_json_lines(root / "yelp_academic_dataset_business.json", max_rows.get("business"))
    r = read_json_lines(root / "yelp_academic_dataset_review.json", max_rows.get("review"))

    b["categories"] = b["categories"].fillna("")
    b["attributes_text"] = b["attributes"].apply(attributes_to_text)
    b["biz_text"] = (
        b["name"].fillna("") + " " + b["categories"] + " " + b["attributes_text"]
    ).apply(normalize_text)

    r["date"] = pd.to_datetime(r["date"], errors="coerce")
    r["text"] = r["text"].fillna("").apply(normalize_text)

    business_df = b[[
        "business_id","name","city","state","latitude","longitude",
        "stars","review_count","is_open","categories","attributes_text","biz_text"
    ]].copy()

    review_df = r[[
        "review_id","user_id","business_id","stars","date","text","useful","funny","cool"
    ]].copy()

    return business_df, review_df


def preprocess(
    business_df: pd.DataFrame,
    review_df: pd.DataFrame,
    *,
    state: Optional[str] = None,
    city: Optional[str] = None,
    category_contains: Optional[str] = None,
    min_biz_reviews: int = 20,
    min_user_reviews: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Standard recommender preprocessing:
    - optional region/category filters
    - prune low-activity users/items
    - build implicit interactions with engagement weights
    """
    b = business_df.copy()
    r = review_df.copy()

    if state:
        b = b[b["state"] == state]
    if city:
        b = b[b["city"].str.lower() == city.lower()]

    if category_contains:
        patt = category_contains.lower()
        b = b[b["categories"].fillna("").str.lower().str.contains(patt, na=False)]

    b = b[b["review_count"] >= min_biz_reviews]
    r = r[r["business_id"].isin(set(b["business_id"]))].copy()

    user_counts = r["user_id"].value_counts()
    good_users = set(user_counts[user_counts >= min_user_reviews].index)
    r = r[r["user_id"].isin(good_users)].copy()

    # engagement-derived implicit weight
    engage = (r["useful"].fillna(0) + r["funny"].fillna(0) + r["cool"].fillna(0)).astype(float)
    weight = 1.0 + np.log1p(engage)

    interactions_df = pd.DataFrame({
        "user_id": r["user_id"].astype(str).values,
        "business_id": r["business_id"].astype(str).values,
        "timestamp": r["date"].values,
        "rating": r["stars"].astype(np.float32).values,
        "weight": weight.astype(np.float32).values,
        "query_text": r["text"].astype(str).values,
    }).dropna(subset=["timestamp"])

    # remove dead businesses after prune
    keep_biz = set(interactions_df["business_id"].value_counts().index)
    b = b[b["business_id"].astype(str).isin(keep_biz)].copy()
    b["business_id"] = b["business_id"].astype(str)

    return b.reset_index(drop=True), interactions_df.reset_index(drop=True)


def leave_last_out(interactions_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Per-user temporal split:
      last interaction -> test
      earlier          -> train
    """
    df = interactions_df.sort_values(["user_id", "timestamp"])
    last_idx = df.groupby("user_id").tail(1).index
    test = df.loc[last_idx].copy()
    train = df.drop(index=last_idx).copy()
    return train.reset_index(drop=True), test.reset_index(drop=True)


def load_business_image_embeddings(path: str) -> pd.DataFrame:
    """
    Load per-business image embedding table produced by scripts/compute_clip_embeddings.py

    Expected columns:
      - business_id (str)
      - image_emb (np.ndarray stored via npy, or columns image_emb_0..image_emb_{d-1})
    This helper supports a simple CSV format with one column per dimension.
    """
    p = Path(path)
    if p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)

    if "business_id" not in df.columns:
        raise ValueError("Embeddings file must contain business_id column.")

    # If stored as multiple columns, keep them as float32 numpy matrix later
    df["business_id"] = df["business_id"].astype(str)
    return df
