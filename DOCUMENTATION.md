# Documentation — Hybrid Retrieval + Ranking + CLIP + Evaluation + Config

This document is blunt and implementation-focused: it describes **what each component does**, **what it expects**, and **how the full pipeline fits together**.

---

## 1) Configuration and Reproducibility

### `modeling/config_utils.py`

- `load_config(path)`
  - loads YAML config into a Python dict

- `set_global_seeds(seed)`
  - sets Python, NumPy, and TensorFlow random seeds
  - reduces run-to-run variance (not full determinism)

- `ensure_dir(path)`
  - creates an output directory if missing

The config drives:
- dataset filtering thresholds
- which modalities are enabled (text, images)
- model dimensions
- training epochs/batch sizes
- retrieval/ranking K values
- output locations

---

## 2) Data Pipeline

### `modeling/data_pipeline.py`

- `read_json_lines(path, max_rows=None)`
  - reads Yelp JSONL into a DataFrame

- `normalize_text(s)`
  - lowercases and normalizes whitespace

- `attributes_to_text(attrs)`
  - flattens nested Yelp attribute dictionaries into strings
  - reason: text encoders need linearized inputs

- `load_yelp(root, max_rows=None)`
  - loads business + review JSONL
  - builds `biz_text = name + categories + attributes_text`
  - normalizes review text and parses timestamps

- `preprocess(business_df, review_df, ...)`
  - optional region and category filtering
  - prunes low-activity businesses and users
  - builds implicit feedback edges:
    - `weight = 1 + log1p(useful + funny + cool)`
  - outputs:
    - filtered `business_df`
    - `interactions_df` with user_id, business_id, timestamp, rating, weight, query_text

- `leave_last_out(interactions_df)`
  - per-user temporal split:
    - last interaction → test
    - earlier interactions → train
  - avoids leakage

- `load_business_image_embeddings(path)`
  - reads a CSV/Parquet produced by `scripts/compute_clip_embeddings.py`
  - expected embedding columns: `image_emb_0..image_emb_{d-1}`

---

## 3) Stage 1 — Retrieval Model

### `modeling/retrieval_model.py` → `YelpRetrievalModel`

Two-tower retrieval model (TFRS Retrieval):

**Query tower**
- user_id → embedding
- optional query_text → TextVectorization → token embedding → pooling
- concat → MLP → L2-normalized query vector

**Candidate tower**
- business_id → embedding
- optional biz_text → TextVectorization → token embedding → pooling
- optional image_emb (precomputed CLIP) → projection MLP
- concat → MLP → L2-normalized candidate vector

Key methods:
- `adapt_text(train_ds, cand_ds)`
  - fits TextVectorization on observed text
- `query_tower(features)`
- `candidate_tower(features)`
- `compute_loss(features)`
  - retrieval task loss with optional engagement-based sample weights

**Why the candidate tower uses offline CLIP**
- avoids mixing TF training with large image models
- image embeddings are stable features consumed like any other vector

---

## 4) Stage 2 — Ranking Model

### `modeling/ranking_model.py` → `YelpRankingModel`

A feature-rich ranker (TFRS Ranking) that re-scores top-N candidates.

Inputs (typical):
- user_id embedding
- business_id embedding
- optional query_text embedding
- optional biz_text embedding
- optional image_emb projection
- numeric features: business stars, review_count (simple priors)

Outputs:
- a scalar relevance score

Training label:
- default: rating regression (MSE + RMSE metric)
- can be changed to binary relevance by transforming `label` in dataset creation

---

## 5) Inference Engine

### `modeling/hybrid_engine.py` → `HybridEngine`

Responsibilities:
- build retrieval index from candidate tower
- retrieve top-N candidate IDs for a (user_id, query_text) pair
- optionally rerank those candidates using the ranking model

Key methods:
- `retrieve(user_id, query_text, k)` → list[business_id]
- `rerank(user_id, query_text, candidate_ids, k)` → DataFrame of top-k scored businesses
- `recommend(user_id, query_text, k)` → retrieve then rerank

---

## 6) CLIP Image Branch (Fully Wired)

### `scripts/compute_clip_embeddings.py`

What it does:
- loads Yelp photo metadata (`yelp_academic_dataset_photo.json`)
- finds local image files by photo_id
- runs OpenCLIP image encoder (PyTorch)
- aggregates per-business embeddings by mean pooling
- writes `business_id, image_emb_0..image_emb_{d-1}`

How it connects to TF models:
- retrieval candidate tower reads `image_emb`
- ranking model reads `image_emb`
- hybrid engine loads the embedding table and supplies vectors at index time and reranking

This is “fully wired” because:
- the code path is end-to-end (precompute → load → model features → indexing → rerank)

---

## 7) Evaluation

### `scripts/metrics.py`
- `recall_at_k(retrieved, positive, k)`
- `ndcg_at_k(retrieved, positive, k)`
  - binary relevance (single held-out positive)

### `scripts/evaluate_retrieval.py`
- trains retrieval model (stage 1)
- builds retrieval index
- evaluates Recall@K and NDCG@K on leave-last-out test items

### `scripts/evaluate_ranking.py`
- trains retrieval model
- trains ranking model
- retrieval → rerank pipeline
- evaluates Recall@K and NDCG@K on reranked output

Important detail:
- evaluation is per-user held-out item; this is the standard minimal RS metric setup.
- if you want graded relevance (e.g., true NDCG using ratings), you need multiple test items/user or sampled negatives.

---

## 8) Experiment Workflow Summary

1) Edit `configs/config.yaml`
2) Run training:
   - `python train.py --config configs/config.yaml`
3) Evaluate retrieval:
   - `python scripts/evaluate_retrieval.py --config configs/config.yaml`
4) (Optional) Compute CLIP embeddings:
   - `python scripts/compute_clip_embeddings.py ...`
5) Enable images in config and rerun ranking eval:
   - `python scripts/evaluate_ranking.py --config configs/config.yaml --image_embeddings outputs/business_image_embeddings.csv`
