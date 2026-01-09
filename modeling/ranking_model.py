from typing import Dict, Any, Optional

import tensorflow as tf
import tensorflow_recommenders as tfrs


class YelpRankingModel(tfrs.Model):
    """
    Second-stage ranker:
      Input: user_id, business_id, (optional) query_text, biz_text, image_emb, plus numeric features
      Output: relevance score (regression on rating or binary like)

    Default label: rating (float). You can switch to binary by transforming labels in the dataset.
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
    ):
        super().__init__()
        self.use_query_text = use_query_text
        self.use_biz_text = use_biz_text
        self.use_image_embeddings = use_image_embeddings
        self.image_embedding_dim = image_embedding_dim

        self.user_lookup = tf.keras.layers.StringLookup(vocabulary=user_ids, mask_token=None)
        self.biz_lookup = tf.keras.layers.StringLookup(vocabulary=business_ids, mask_token=None)

        self.user_emb = tf.keras.layers.Embedding(self.user_lookup.vocabulary_size(), embedding_dim)
        self.biz_emb = tf.keras.layers.Embedding(self.biz_lookup.vocabulary_size(), embedding_dim)

        self.query_vec = tf.keras.layers.TextVectorization(
            max_tokens=max_tokens, output_mode="int", output_sequence_length=query_seq_len
        )
        self.biz_vec = tf.keras.layers.TextVectorization(
            max_tokens=max_tokens, output_mode="int", output_sequence_length=biz_seq_len
        )

        self.query_tok_emb = tf.keras.layers.Embedding(max_tokens, embedding_dim)
        self.biz_tok_emb = tf.keras.layers.Embedding(max_tokens, embedding_dim)

        self.query_pool = tf.keras.layers.GlobalAveragePooling1D()
        self.biz_pool = tf.keras.layers.GlobalAveragePooling1D()

        if self.use_image_embeddings:
            self.img_proj = tf.keras.Sequential([
                tf.keras.layers.Dense(mlp_hidden, activation="relu"),
                tf.keras.layers.Dense(embedding_dim),
            ])
        else:
            self.img_proj = None

        # Ranker head
        self.ranker = tf.keras.Sequential([
            tf.keras.layers.Dense(mlp_hidden, activation="relu"),
            tf.keras.layers.Dense(mlp_hidden, activation="relu"),
            tf.keras.layers.Dense(1),
        ])

        # Regression on rating by default
        self.task = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")],
        )

    def adapt_text(self, train_ds: tf.data.Dataset, cand_ds: tf.data.Dataset) -> None:
        if self.use_query_text and "query_text" in train_ds.element_spec:
            self.query_vec.adapt(train_ds.map(lambda x: x["query_text"]))
        if self.use_biz_text and "biz_text" in cand_ds.element_spec:
            self.biz_vec.adapt(cand_ds.map(lambda x: x["biz_text"]))

    def call(self, features: Dict[str, Any]) -> tf.Tensor:
        parts = []

        u = self.user_emb(self.user_lookup(features["user_id"]))
        b = self.biz_emb(self.biz_lookup(features["business_id"]))
        parts.extend([u, b])

        if self.use_query_text:
            qt = self.query_vec(features.get("query_text", tf.constant([""])))
            q = self.query_pool(self.query_tok_emb(qt))
            parts.append(q)

        if self.use_biz_text:
            bt = self.biz_vec(features.get("biz_text", tf.constant([""])))
            t = self.biz_pool(self.biz_tok_emb(bt))
            parts.append(t)

        if self.use_image_embeddings:
            img = tf.cast(features["image_emb"], tf.float32)
            parts.append(self.img_proj(img))

        # numeric side features (optional; pass 0 if missing)
        for name in ["biz_stars", "biz_review_count"]:
            if name in features:
                x = tf.cast(tf.expand_dims(features[name], -1), tf.float32)
                parts.append(x)

        x = tf.concat(parts, axis=-1)
        return self.ranker(x)

    def compute_loss(self, features: Dict[str, Any], training: bool = False) -> tf.Tensor:
        labels = tf.cast(features["label"], tf.float32)
        preds = self(features)
        return self.task(labels=labels, predictions=preds)
