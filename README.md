# Hybrid Search + Recommendation System (TensorFlow / TFRS) — with Ranking + CLIP + Eval + Config

This repo implements a **two-stage hybrid retrieval system** on the **Yelp Open Dataset**:

- **Stage 1 (Retrieval):** Two-tower model (TFRS Retrieval)
- **Stage 2 (Ranking):** Feature-rich ranker (TFRS Ranking) re-scores top-N candidates
- **Optional Multimodal:** Precomputed **CLIP image embeddings** wired into candidate and ranking models
- **Evaluation:** Recall@K and NDCG@K scripts for retrieval-only and reranked output
- **Reproducibility:** YAML-based experiment config + global seeding

## What “Hybrid” means here
You can run:
- **Search-only:** `user_id="ANON"` + `query_text="ramen"`
- **Rec-only:** `user_id=REAL_USER` + `query_text=""`
- **Hybrid:** `user_id=REAL_USER` + `query_text="late night coffee"`

The query tower mixes user embedding + query text embedding; candidate tower mixes business id + business text (+ optional image embedding).

## Project structure
```
hybrid-yelp-recsys/
├── README.md
├── DOCUMENTATION.md
├── requirements.txt
├── train.py
├── configs/
│   └── config.yaml
├── modeling/
│   ├── config_utils.py
│   ├── data_pipeline.py
│   ├── retrieval_model.py
│   ├── ranking_model.py
│   └── hybrid_engine.py
└── scripts/
    ├── compute_clip_embeddings.py
    ├── evaluate_retrieval.py
    ├── evaluate_ranking.py
    └── metrics.py
```

## Setup
```bash
pip install -r requirements.txt
```

Put Yelp files here:
```
data/yelp/
  yelp_academic_dataset_business.json
  yelp_academic_dataset_review.json
  (optional) yelp_academic_dataset_photo.json
```

## Run: Retrieval + Ranking training (default)
```bash
python train.py --config configs/config.yaml
```

Outputs are written under `experiment.output_dir` in the config.

## Evaluation
Retrieval-only (stage 1):
```bash
python scripts/evaluate_retrieval.py --config configs/config.yaml
```

Full pipeline (retrieval → rerank):
```bash
python scripts/evaluate_ranking.py --config configs/config.yaml
```

## CLIP image branch (optional)
1) Download Yelp photos separately (Yelp provides photos as a separate artifact).
2) Precompute per-business CLIP embeddings:
```bash
python scripts/compute_clip_embeddings.py   --photo_json data/yelp/yelp_academic_dataset_photo.json   --images_dir /path/to/yelp_photos   --out_csv outputs/business_image_embeddings.csv
```

3) Enable images in `configs/config.yaml`:
```yaml
features:
  use_image_embeddings: true
```
4) Run training/eval with:
```bash
python train.py --config configs/config.yaml --image_embeddings outputs/business_image_embeddings.csv
python scripts/evaluate_ranking.py --config configs/config.yaml --image_embeddings outputs/business_image_embeddings.csv
```

## Notes / limitations (intentional)
- The CLIP step is offline precompute to keep TF training simple and fast.
- The evaluation uses **leave-last-out** (per-user) with binary relevance (held-out item).
- This is research-prototype quality: the interfaces are clean, but production hardening is out of scope.
