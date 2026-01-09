from typing import Optional, Dict, Any

import tensorflow as tf
import tensorflow_recommenders as tfrs


class YelpRetrievalModel(tfrs.Model):
    """
    Two-tower retrieval model (TFRS):
      - Query tower: user_id embedding + query_text embedding
      - Candidate tower: business_id embedding + biz_text embedding (+ optional image embedding)
    """
    def __init__(
        self,
        *,
        user_ids,
        business_ids,
        embedding_dim: int = 64,
        max_tokens: int = 50_000,
        query_seq_len: int = 64,
        biz_seq_len: int = 128,
        mlp_hidden: int = 128,
        use_query_text: bool = True,
        use_biz_text: bool = True,
        use_image_embeddings: bool = False,
        image_embedding_dim: int = 512,
        candidates_ds: Optional[tf.data.Dataset] = None,
        top_k: int = 100,
    ):
        super().__init__()
        self.use_query_text = use_query_text
        self.use_biz_text = use_biz_text
        self.use_image_embeddings = use_image_embeddings
        self.image_embedding_dim = image_embedding_dim

        # ID vocab
        self.user_lookup = tf.keras.layers.StringLookup(vocabulary=user_ids, mask_token=None)
        self.biz_lookup = tf.keras.layers.StringLookup(vocabulary=business_ids, mask_token=None)

        self.user_emb = tf.keras.layers.Embedding(self.user_lookup.vocabulary_size(), embedding_dim)
        self.biz_id_emb = tf.keras.layers.Embedding(self.biz_lookup.vocabulary_size(), embedding_dim)

        # Text vectorizers
        self.query_vec = tf.keras.layers.TextVectorization(
            max_tokens=max_tokens,
            output_mode="int",
            output_sequence_length=query_seq_len,
        )
        self.biz_vec = tf.keras.layers.TextVectorization(
            max_tokens=max_tokens,
            output_mode="int",
            output_sequence_length=biz_seq_len,
        )
        self.query_tok_emb = tf.keras.layers.Embedding(max_tokens, embedding_dim)
        self.biz_tok_emb = tf.keras.layers.Embedding(max_tokens, embedding_dim)

        self.query_pool = tf.keras.layers.GlobalAveragePooling1D()
        self.biz_pool = tf.keras.layers.GlobalAveragePooling1D()

        # Optional image projection (precomputed CLIP embedding -> embedding_dim)
        if self.use_image_embeddings:
            self.img_proj = tf.keras.Sequential([
                tf.keras.layers.Dense(mlp_hidden, activation="relu"),
                tf.keras.layers.Dense(embedding_dim),
            ])
        else:
            self.img_proj = None

        # MLP heads
        self.query_mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(mlp_hidden, activation="relu"),
            tf.keras.layers.Dense(embedding_dim),
        ])
        self.cand_mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(mlp_hidden, activation="relu"),
            tf.keras.layers.Dense(embedding_dim),
        ])

        # Retrieval task (with optional FactorizedTopK metric if candidates_ds provided)
        metrics = None
        if candidates_ds is not None:
            metrics = tfrs.metrics.FactorizedTopK(
                candidates=candidates_ds.map(lambda x: self.candidate_tower(x)).batch(4096)
                if hasattr(candidates_ds, "map") else candidates_ds
            )
        self.task = tfrs.tasks.Retrieval(metrics=metrics)

        self._top_k = top_k

    def adapt_text(self, train_ds: tf.data.Dataset, cand_ds: tf.data.Dataset) -> None:
        if self.use_query_text:
            self.query_vec.adapt(train_ds.map(lambda x: x["query_text"]))
        if self.use_biz_text:
            self.biz_vec.adapt(cand_ds.map(lambda x: x["biz_text"]))

    def query_tower(self, features: Dict[str, Any]) -> tf.Tensor:
        u = self.user_emb(self.user_lookup(features["user_id"]))

        parts = [u]

        if self.use_query_text:
            qt = self.query_vec(features["query_text"])
            q = self.query_pool(self.query_tok_emb(qt))
            parts.append(q)

        x = tf.concat(parts, axis=-1) if len(parts) > 1 else parts[0]
        x = self.query_mlp(x)
        return tf.math.l2_normalize(x, axis=-1)

    def candidate_tower(self, features: Dict[str, Any]) -> tf.Tensor:
        b = self.biz_id_emb(self.biz_lookup(features["business_id"]))

        parts = [b]

        if self.use_biz_text:
            bt = self.biz_vec(features["biz_text"])
            t = self.biz_pool(self.biz_tok_emb(bt))
            parts.append(t)

        if self.use_image_embeddings:
            # Expect feature "image_emb" shape [batch, image_embedding_dim]
            img = tf.cast(features["image_emb"], tf.float32)
            parts.append(self.img_proj(img))

        x = tf.concat(parts, axis=-1) if len(parts) > 1 else parts[0]
        x = self.cand_mlp(x)
        return tf.math.l2_normalize(x, axis=-1)

    def compute_loss(self, features: Dict[str, Any], training: bool = False) -> tf.Tensor:
        q = self.query_tower(features)

        # Positive candidate embedding: use only business_id embedding for positive label during training
        # (keeps training cheap; indexing uses full candidate tower with text/images).
        pos = tf.math.l2_normalize(self.biz_id_emb(self.biz_lookup(features["business_id"])), axis=-1)

        return self.task(q, pos, sample_weight=features.get("weight", None))
