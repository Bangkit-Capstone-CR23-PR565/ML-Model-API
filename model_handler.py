import tensorflow as tf
import df_loader
import numpy as np

ranking_model_path = "./ranking_model"
retrieval_model_path = "./retrieval_model"

# Get most relevant items
# Should return score, id, event_name
def retrieval_model(user_id):
    loaded = tf.saved_model.load(retrieval_model_path)
    scores, event_ids = loaded(tf.constant([user_id]))
    scores = [i.numpy() for i in scores[0]]
    event_ids = [i.numpy() for i in event_ids[0]]

    # scores and titles should have same size
    output = []
    for (event_id,score) in zip(event_ids,scores):
        output.append({
            'id': int(event_id),
            'relevancy_score': float(score)
        })
    return output

# Give items a rankable value
# Should return event_id, event_name, rating
def ranking_model(user_id):
    events_df = df_loader.get_events_df()
    loaded = tf.saved_model.load(ranking_model_path)
    model_rating_predictions = loaded(
        {
            "event_id": tf.constant(events_df['id'].to_list(), dtype=tf.int64),
            "name": tf.constant(events_df['name'].to_list(), dtype=tf.string),
            "category": tf.constant(events_df['category'].to_list(), dtype=tf.string),
            "user_id": tf.constant([user_id]*len(events_df.index), dtype=tf.int64)
        }
    )
    ratings = [{
        'id':int(id),
        'rating_prediction':float(rating[0].numpy())
        } for id,rating in zip(events_df['id'].to_list(),model_rating_predictions)]
    return sorted(ratings, key=lambda x: x['rating_prediction'], reverse=True)

def tags_search_model(query, top_n):
    events_df = df_loader.get_events_df()
    if top_n == None:
        top_n = len(events_df)
    
    # to remove string quotes correctly, we need to parse data into list first, then join it back into string
    event_tags_list = [value if isinstance(value, str) else '' for value in events_df['category']]
    event_name_list = [value if isinstance(value, str) else '' for value in events_df['name']]
    event_location_list = [value if isinstance(value, str) else '' for value in events_df['location']]
    event_description_list = [value if isinstance(value, str) else '' for value in events_df['description']]

    data = event_tags_list + event_name_list + event_location_list + event_description_list

    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=10_000,
        output_mode='tf_idf',
    )

    vectorizer.adapt(data)
    X = vectorizer(data+[query])

    cosine_sim = tf.linalg.matmul(X, X, transpose_b=True)
    sim_scores = list(enumerate(cosine_sim[-1, :-1]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_docs = sim_scores[:top_n]

    # handle multiple event with same list of tags
    output = []
    c=0
    read_index_data = set()
    for index, score in top_docs:
        if data[index] in read_index_data:
            continue
        print(read_index_data)
        if float(score) == 0:
            break
        matched_events = events_df[
            (events_df['category']==data[index]) |
            (events_df['name']==data[index]) |
            (events_df['location']==data[index]) |
            (events_df['description']==data[index])
            ]
        ids = [value if not np.isnan(value) else '' for value in matched_events['id']]
        for i in range(len(ids)):
            if c >= top_n:
                break
            output.append({
                'id': ids[i],
                'match_score': float(score)
            })
            c += 1
        read_index_data.add(data[index])
    return output