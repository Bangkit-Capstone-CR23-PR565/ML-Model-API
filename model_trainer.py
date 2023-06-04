from typing import Dict, Text
import tensorflow as tf
import tensorflow_recommenders as tfrs
import numpy as np
import df_loader

retrieval_model_path = "./retrieval_model"
ranking_model_path = "./ranking_model"


def retrain_all():
    processed_df = df_loader.get_processed_df()
    
    ratings = tf.data.Dataset.from_tensor_slices(dict(processed_df))
    events = tf.data.Dataset.from_tensor_slices(dict(processed_df)).map(lambda x: x["event_id"])
    
    shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)
    n = tf.data.experimental.cardinality(shuffled)

    train = shuffled.take(int(n*4/5))
    test = shuffled.skip(int(n*4/5)).take(int(n*1/5))
    cached_train = train.shuffle(100_000).batch(8192).cache()
    cached_test = test.batch(4096).cache()

    event_ids = events.batch(1_000)
    user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"])
    event_names = ratings.batch(1_000_000).map(lambda x: x["name"])
    categories = ratings.batch(1_000_000).map(lambda x: x["category"])

    unique_event_ids = np.unique(np.concatenate(list(event_ids)))
    unique_event_names = np.unique(np.concatenate(list(event_names)))
    unique_categories = np.unique(np.concatenate(list(categories)))
    unique_user_ids = np.unique(np.concatenate(list(user_ids)))

    class SubRankingModel(tf.keras.Model):
        def __init__(self):
            super().__init__()
            embedding_dimension = 32
            max_tokens=10_000
            self.user_embeddings = tf.keras.Sequential([
                tf.keras.layers.IntegerLookup(
                    vocabulary=unique_user_ids, mask_token=None),
                tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
            ])
            self.event_embeddings = tf.keras.Sequential([
                tf.keras.layers.IntegerLookup(
                    vocabulary=unique_event_ids, mask_token=None),
                tf.keras.layers.Embedding(len(unique_event_ids) + 1, embedding_dimension)
            ])
            self.title_vectorizer = tf.keras.layers.TextVectorization(max_tokens=max_tokens)
            self.title_text_embedding = tf.keras.Sequential([
                self.title_vectorizer,
                tf.keras.layers.Embedding(max_tokens, embedding_dimension, mask_zero=True),
                tf.keras.layers.GlobalAveragePooling1D()
            ])
            self.title_vectorizer.adapt(unique_event_names)
            self.category_vectorizer = tf.keras.layers.TextVectorization(max_tokens=max_tokens,)
            self.category_text_embedding = tf.keras.Sequential([
                self.category_vectorizer,
                tf.keras.layers.Embedding(max_tokens, embedding_dimension, mask_zero=True),
                tf.keras.layers.GlobalAveragePooling1D()
            ])
            self.category_vectorizer.adapt(unique_categories)
            self.ratings = tf.keras.Sequential([
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(1)
            ])
        def call(self, inputs):
            user_id, event_id, event_name, category = inputs
            user_embedding = self.user_embeddings(user_id)
            event_embedding = self.event_embeddings(event_id)
            title_text_embedding = self.title_text_embedding(event_name)
            category_text_embeddings = self.category_text_embedding(category)
            return self.ratings(tf.concat([
                user_embedding,
                event_embedding,
                title_text_embedding,
                category_text_embeddings], axis=1))

    class RankingModel(tfrs.models.Model):
        def __init__(self):
            super().__init__()
            self.subranking_model = SubRankingModel()
            self.rating_task = tfrs.tasks.Ranking(
                loss=tf.keras.losses.MeanSquaredError(),
                metrics=[tf.keras.metrics.RootMeanSquaredError()]
            )
        def call(self, features):
            rating_predictions = self.subranking_model((features['user_id'],features['event_id'],features['name'],features['category']))
            return rating_predictions
        def compute_loss(self, features, training=False):
            ratings=features.pop("user_rating")
            rating_predictions = self(features)
            rating_loss = self.rating_task(labels=ratings, predictions=rating_predictions)
            return rating_loss

    class RetrievalModel(tfrs.Model):
        def __init__(self):
            super().__init__()
            embedding_dimension = 32
            self.user_embeddings = tf.keras.Sequential([
                tf.keras.layers.IntegerLookup(
                    vocabulary=unique_user_ids, mask_token=None),
                tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
            ])
            self.event_embeddings = tf.keras.Sequential([
                tf.keras.layers.IntegerLookup(
                    vocabulary=unique_event_ids, mask_token=None),
                tf.keras.layers.Embedding(len(unique_event_ids) + 1, embedding_dimension)
            ])
            self.retrieval_task = tfrs.tasks.Retrieval(
                metrics=tfrs.metrics.FactorizedTopK(
                    candidates=events.batch(128).map(self.event_embeddings)
                )
            )
        def compute_loss(self, features, training=False):
            user_embeddings = self.user_embeddings(features['user_id'])
            event_embeddings = self.event_embeddings(features['event_id'])
            return self.retrieval_task(user_embeddings, event_embeddings)
    
    # Fit and save retrieval model
    retrieval_model = RetrievalModel()
    retrieval_model.compile(optimizer=tf.keras.optimizers.Adadelta(0.05))
    retrieval_model.fit(cached_train, epochs=5, verbose=0)
    index = tfrs.layers.factorized_top_k.BruteForce(retrieval_model.user_embeddings, k=len(processed_df))
    index.index_from_dataset(
        tf.data.Dataset.zip((events.batch(100), events.batch(100).map(retrieval_model.event_embeddings)))
    )
    index(tf.constant([1]))
    tf.saved_model.save(index, retrieval_model_path)

    # Fit and save ranking model
    ranking_model = RankingModel()
    ranking_model.compile(optimizer=tf.keras.optimizers.SGD(0.1))
    ranking_model.fit(cached_train, epochs=7, verbose=0)
    ranking_model({
        "user_id": tf.constant([1]),
        "event_id": tf.constant([2]),
        "name": tf.constant([""]),
        "category": tf.constant([""]),
        })
    tf.saved_model.save(ranking_model, ranking_model_path)
    return "Training finished"