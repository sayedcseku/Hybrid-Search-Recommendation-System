import argparse
import os
import numpy as np
import pandas as pd
import tensorflow as tf

from modeling.config_utils import load_config, set_global_seeds, ensure_dir
from modeling.data_pipeline import load_yelp, preprocess, leave_last_out, load_business_image_embeddings
from modeling.retrieval_model import YelpRetrievalModel
from modeling.ranking_model import YelpRankingModel
from modeling.hybrid_engine import HybridEngine


def build_tf_datasets(train_df: pd.DataFrame, business_df: pd.DataFrame, batch_size: int,
                      use_image: bool, img_df=None):
    train_ds = tf.data.Dataset.from_tensor_slices({
        "user_id": train_df["user_id"].astype(str).values,
        "business_id": train_df["business_id"].astype(str).values,
        "query_text": train_df["query_text"].astype(str).values,
        "weight": train_df["weight"].astype("float32").values,
    }).shuffle(100_000, seed=42, reshuffle_each_iteration=True).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Candidate ds for adapting text and indexing
    cand = {
        "business_id": business_df["business_id"].astype(str).values,
        "biz_text": business_df["biz_text"].astype(str).values,
    }

    if use_image:
        emb_cols = [c for c in img_df.columns if c.startswith("image_emb_")]
        image_map = dict(zip(img_df["business_id"].astype(str).tolist(), img_df[emb_cols].astype("float32").to_numpy()))
        img = np.stack([image_map.get(bid, np.zeros(len(emb_cols), dtype="float32")) for bid in cand["business_id"]], axis=0)
        cand["image_emb"] = img

    cand_ds = tf.data.Dataset.from_tensor_slices(cand).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return train_ds, cand_ds


def build_rank_ds(train_df: pd.DataFrame, business_df: pd.DataFrame, batch_size: int,
                  use_image: bool, img_df=None):
    biz = business_df[["business_id","biz_text","stars","review_count"]].copy()
    df = train_df.merge(biz, on="business_id", how="left")
    df["label"] = df["rating"].astype("float32").fillna(0)

    rank_features = {
        "user_id": df["user_id"].astype(str).values,
        "business_id": df["business_id"].astype(str).values,
        "query_text": df["query_text"].astype(str).values,
        "biz_text": df["biz_text"].astype(str).fillna("").values,
        "biz_stars": df["stars"].astype("float32").fillna(0).values,
        "biz_review_count": df["review_count"].astype("float32").fillna(0).values,
        "label": df["label"].values,
    }
    if use_image:
        emb_cols = [c for c in img_df.columns if c.startswith("image_emb_")]
        df = df.merge(img_df[["business_id"] + emb_cols], on="business_id", how="left")
        for c in emb_cols:
            df[c] = df[c].astype("float32").fillna(0.0)
        rank_features["image_emb"] = df[emb_cols].astype("float32").to_numpy()

    rank_ds = tf.data.Dataset.from_tensor_slices(rank_features).shuffle(200_000, seed=42, reshuffle_each_iteration=True)\
        .batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return rank_ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--image_embeddings", default=None, help="CSV/Parquet with business_id + image_emb_0..")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_global_seeds(int(cfg["experiment"]["seed"]))
    out_dir = cfg["experiment"]["output_dir"]
    ensure_dir(out_dir)

    # Data
    business_df, review_df = load_yelp(cfg["data"]["root"])
    business_df, inter = preprocess(
        business_df, review_df,
        state=cfg["data"]["state"],
        city=cfg["data"]["city"],
        category_contains=cfg["data"]["category_contains"],
        min_biz_reviews=int(cfg["data"]["min_biz_reviews"]),
        min_user_reviews=int(cfg["data"]["min_user_reviews"]),
    )
    train_df, test_df = leave_last_out(inter)

    use_image = bool(cfg["features"]["use_image_embeddings"])
    img_df = None
    if use_image:
        if not args.image_embeddings:
            raise ValueError("Config sets use_image_embeddings=true but --image_embeddings not provided.")
        img_df = load_business_image_embeddings(args.image_embeddings)

    batch_size = int(cfg["training"]["batch_size"])

    # TF datasets
    train_ds, cand_ds = build_tf_datasets(train_df, business_df, batch_size, use_image, img_df)
    rank_ds = build_rank_ds(train_df, business_df, batch_size, use_image, img_df)

    # Vocabularies
    user_ids = train_df["user_id"].astype(str).unique().tolist()
    business_ids = business_df["business_id"].astype(str).unique().tolist()

    # Stage 1: retrieval
    retrieval = YelpRetrievalModel(
        user_ids=user_ids,
        business_ids=business_ids,
        embedding_dim=int(cfg["model"]["embedding_dim"]),
        max_tokens=int(cfg["model"]["max_tokens"]),
        query_seq_len=int(cfg["model"]["query_seq_len"]),
        biz_seq_len=int(cfg["model"]["biz_seq_len"]),
        mlp_hidden=int(cfg["model"]["mlp_hidden"]),
        use_query_text=bool(cfg["features"]["use_query_text"]),
        use_biz_text=bool(cfg["features"]["use_biz_text"]),
        use_image_embeddings=use_image,
        image_embedding_dim=int(cfg["features"]["image_embedding_dim"]),
        candidates_ds=None,
        top_k=int(cfg["retrieval"]["top_k"]),
    )
    retrieval.adapt_text(train_ds, cand_ds)
    retrieval.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=float(cfg["training"]["learning_rate"])))
    retrieval.fit(train_ds, epochs=int(cfg["training"]["epochs_retrieval"]), verbose=1)

    # Stage 2: ranking
    ranker = YelpRankingModel(
        user_ids=user_ids,
        business_ids=business_ids,
        embedding_dim=int(cfg["model"]["embedding_dim"]),
        max_tokens=int(cfg["model"]["max_tokens"]),
        query_seq_len=int(cfg["model"]["query_seq_len"]),
        biz_seq_len=int(cfg["model"]["biz_seq_len"]),
        mlp_hidden=int(cfg["model"]["mlp_hidden"]),
        use_query_text=bool(cfg["features"]["use_query_text"]),
        use_biz_text=bool(cfg["features"]["use_biz_text"]),
        use_image_embeddings=use_image,
        image_embedding_dim=int(cfg["features"]["image_embedding_dim"]),
    )
    ranker.adapt_text(rank_ds, cand_ds)
    ranker.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=float(cfg["training"]["learning_rate"])))
    ranker.fit(rank_ds, epochs=int(cfg["training"]["epochs_ranking"]), verbose=1)

    # Save models
    retrieval.save(os.path.join(out_dir, "retrieval_model"))
    ranker.save(os.path.join(out_dir, "ranking_model"))

    # Build engine and save a small demo output
    engine = HybridEngine(
        retrieval_model=retrieval,
        business_df=business_df,
        use_biz_text=bool(cfg["features"]["use_biz_text"]),
        use_image_embeddings=use_image,
        image_embeddings_df=img_df,
        top_k_retrieval=int(cfg["retrieval"]["top_k"]),
        ranking_model=ranker,
    )

    demo_user = train_df["user_id"].iloc[0]
    recs = engine.recommend(user_id=str(demo_user), query_text="", k=10)
    recs.to_csv(os.path.join(out_dir, "demo_recommendations.csv"), index=False)
    print(f"Wrote demo recommendations to {os.path.join(out_dir, 'demo_recommendations.csv')}")

if __name__ == "__main__":
    main()
