from typing import Dict, Text
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs
import numpy as np
import df_loader

retrieval_model_path = "./retrieval_model"
ranking_model_path = "./ranking_model"
events_csv_path = './events.csv'
ratings_csv_path = './ratings.csv'


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

    event_names = events.batch(1_000)
    user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"])

    unique_event_ids = np.unique(np.concatenate(list(event_names)))
    unique_user_ids = np.unique(np.concatenate(list(user_ids)))

    class RetrievalModel(tfrs.Model):
        embedding_dimension = 32
        def __init__(self):
            super().__init__()
            self.event_model: tf.keras.Model = tf.keras.Sequential([
                tf.keras.layers.IntegerLookup(vocabulary=unique_event_ids, mask_token=None),
                tf.keras.layers.Embedding(len(unique_event_ids) + 1, self.embedding_dimension)
            ])
            self.user_model: tf.keras.Model = tf.keras.Sequential([
                tf.keras.layers.IntegerLookup(vocabulary=unique_user_ids, mask_token=None),
                tf.keras.layers.Embedding(len(unique_user_ids) + 1, self.embedding_dimension)
            ])
            self.task: tf.keras.layers.Layer = tfrs.tasks.Retrieval(
            metrics = tfrs.metrics.FactorizedTopK(
                candidates=events.batch(128).map(self.event_model)
                ))

        def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
            user_embeddings = self.user_model(features["user_id"])
            positive_event_embeddings = self.event_model(features["event_id"])

            # parameter: query embedding, candidate embedding.
            return self.task(user_embeddings, positive_event_embeddings)

    class RankingModel(tf.keras.Model):
        embedding_dimension = 32
        def __init__(self):
            super().__init__()
            self.user_embeddings = tf.keras.Sequential([
                tf.keras.layers.IntegerLookup(vocabulary=unique_user_ids, mask_token=None),
                tf.keras.layers.Embedding(len(unique_user_ids) + 1, self.embedding_dimension)
            ])
            self.event_embeddings = tf.keras.Sequential([
                tf.keras.layers.IntegerLookup(vocabulary=unique_event_ids, mask_token=None),
                tf.keras.layers.Embedding(len(unique_event_ids) + 1, self.embedding_dimension)
            ])

            self.ratings = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1)
        ])
        
        def call(self, inputs):
            user_id, event_id = inputs

            user_embedding = self.user_embeddings(user_id)
            event_embedding = self.event_embeddings(event_id)

            # predict rating that the user would give to the event
            return self.ratings(tf.concat([user_embedding, event_embedding], axis=1))
        
    class EventModel(tfrs.models.Model):
        def __init__(self):
            super().__init__()
            self.ranking_model: tf.keras.Model = RankingModel()
            self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
                loss = tf.keras.losses.MeanSquaredError(),
                metrics=[tf.keras.metrics.RootMeanSquaredError()]
            )

        # Call what model to use when making prediction
        def call(self, features: Dict[str, tf.Tensor]) -> tf.Tensor:
            return self.ranking_model(
                (features["user_id"], features["event_id"]))

        def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
            # pop rating as the target label
            labels = features.pop("user_rating")
            
            rating_predictions = self(features)

            # The task computes the loss and the metrics.
            return self.task(labels=labels, predictions=rating_predictions)
    
    # Fit and save retrieval model
    retrieval_model = RetrievalModel()
    retrieval_model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=0.1))
    retrieval_model.fit(cached_train, epochs=20, verbose=0)
    index = tfrs.layers.factorized_top_k.BruteForce(retrieval_model.user_model, k=len(processed_df))
    index.index_from_dataset(
        tf.data.Dataset.zip((events.batch(100), events.batch(100).map(retrieval_model.event_model)))
    )
    index(tf.constant([1]))
    tf.saved_model.save(index, retrieval_model_path)

    # Fit and save ranking model
    ranking_model = EventModel()
    ranking_model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.025))
    ranking_model.fit(cached_train, epochs=10, verbose=0)
    ranking_model({"user_id": tf.constant([1]), "event_id": tf.constant([2])})
    tf.saved_model.save(ranking_model, ranking_model_path)
    return "Training finished"