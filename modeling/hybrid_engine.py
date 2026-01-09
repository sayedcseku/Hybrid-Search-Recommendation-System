from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs


class HybridEngine:
    """
    Inference engine:
      - retrieval stage: get top-N candidates (business_id)
      - ranking stage (optional): rerank those candidates
    """
    def __init__(
        self,
        *,
        retrieval_model,
        business_df: pd.DataFrame,
        use_biz_text: bool = True,
        use_image_embeddings: bool = False,
        image_embeddings_df: Optional[pd.DataFrame] = None,
        top_k_retrieval: int = 200,
        ranking_model=None,
    ):
        self.retrieval_model = retrieval_model
        self.ranking_model = ranking_model
        self.top_k_retrieval = int(top_k_retrieval)

        self.business_df = business_df.copy()
        self.business_df["business_id"] = self.business_df["business_id"].astype(str)

        self.use_biz_text = use_biz_text
        self.use_image_embeddings = use_image_embeddings

        self.image_map = None
        if self.use_image_embeddings:
            if image_embeddings_df is None:
                raise ValueError("use_image_embeddings=True requires image_embeddings_df")
            # Expect columns image_emb_0..image_emb_{d-1}
            emb_cols = [c for c in image_embeddings_df.columns if c.startswith("image_emb_")]
            if not emb_cols:
                raise ValueError("image_embeddings_df must have columns image_emb_0..image_emb_{d-1}")
            self.image_dim = len(emb_cols)
            self.image_map = dict(
                zip(image_embeddings_df["business_id"].astype(str).tolist(),
                    image_embeddings_df[emb_cols].astype(np.float32).to_numpy())
            )
        else:
            self.image_dim = 0

        # Candidate dataset for indexing (must match retrieval candidate_tower inputs)
        cand_features = {
            "business_id": self.business_df["business_id"].astype(str).values,
            "biz_text": self.business_df["biz_text"].astype(str).values,
        }
        if self.use_image_embeddings:
            img = np.stack([self.image_map.get(bid, np.zeros(self.image_dim, dtype=np.float32))
                            for bid in cand_features["business_id"]], axis=0)
            cand_features["image_emb"] = img

        cand_ds = tf.data.Dataset.from_tensor_slices(cand_features).batch(2048)

        # Retrieval index
        self.index = tfrs.layers.factorized_top_k.BruteForce(
            query_model=self.retrieval_model.query_tower,
            k=self.top_k_retrieval
        )
        self.index.index_from_dataset(
            cand_ds.map(lambda x: (x["business_id"], self.retrieval_model.candidate_tower(x)))
        )

    def retrieve(self, *, user_id: str, query_text: str = "", k: Optional[int] = None) -> List[str]:
        k = int(k or self.top_k_retrieval)
        scores, ids = self.index({
            "user_id": tf.constant([str(user_id)]),
            "query_text": tf.constant([str(query_text)]),
        })
        out_ids = [x.decode("utf-8") for x in ids[0].numpy()][:k]
        return out_ids

    def rerank(self, *, user_id: str, query_text: str, candidate_ids: List[str], k: int = 10) -> pd.DataFrame:
        if self.ranking_model is None:
            # no ranker: just return top-k retrieved
            out = self.business_df[self.business_df["business_id"].isin(candidate_ids)].copy()
            out["rank"] = out["business_id"].apply(lambda x: candidate_ids.index(x))
            return out.sort_values("rank").head(k)

        # Build ranker batch
        rows = self.business_df[self.business_df["business_id"].isin(candidate_ids)].copy()
        rows["rank"] = rows["business_id"].apply(lambda x: candidate_ids.index(x))
        rows = rows.sort_values("rank")

        feats = {
            "user_id": tf.constant([str(user_id)] * len(rows)),
            "business_id": tf.constant(rows["business_id"].astype(str).tolist()),
            "query_text": tf.constant([str(query_text)] * len(rows)),
            "biz_text": tf.constant(rows["biz_text"].astype(str).tolist()),
            "biz_stars": tf.constant(rows["stars"].astype(np.float32).fillna(0).to_numpy()),
            "biz_review_count": tf.constant(rows["review_count"].astype(np.float32).fillna(0).to_numpy()),
        }
        if self.use_image_embeddings:
            img = np.stack([self.image_map.get(bid, np.zeros(self.image_dim, dtype=np.float32))
                            for bid in rows["business_id"].astype(str).tolist()], axis=0)
            feats["image_emb"] = tf.constant(img)

        scores = self.ranking_model(feats)
        scores = tf.squeeze(scores, axis=-1).numpy()

        rows["rank_score"] = scores
        rows = rows.sort_values("rank_score", ascending=False).head(k)
        return rows[[
            "business_id","name","city","state","categories","stars","review_count","rank_score"
        ]]

    def recommend(self, *, user_id: str, query_text: str = "", k: int = 10) -> pd.DataFrame:
        cands = self.retrieve(user_id=user_id, query_text=query_text)
        return self.rerank(user_id=user_id, query_text=query_text, candidate_ids=cands, k=k)
