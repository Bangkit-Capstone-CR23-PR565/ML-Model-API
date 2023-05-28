import ast
import tensorflow as tf
import df_loader

ranking_model_path = "./ranking_model"
retrieval_model_path = "./retrieval_model"

# Get most relevant items
# Should return score, id, event_name
def retrieval_model(user_id):
    events_df = df_loader.get_events_df()
    loaded = tf.saved_model.load(retrieval_model_path)
    scores, event_ids = loaded([user_id])
    
    scores = [i.numpy() for i in scores[0]]
    event_ids = [i.numpy() for i in event_ids[0]]

    # scores and titles should have same size
    output = {}
    for i in range(len(scores)):
        event_name = list(events_df[events_df['id']==event_ids[i]]['event_name'])[0]
        output[i] = {
            'event_id': int(event_ids[i]),
            'event_name': str(event_name),
            'relevancy_score': float(scores[i])
        }
    return output

# Give items a rankable value
# Should return event_id, event_name, rating
def ranking_model(user_id):
    events_df = df_loader.get_events_df()
    loaded = tf.saved_model.load(ranking_model_path)
    ratings = {}
    event_ids = list(events_df['id'])
    for event_id in event_ids:
        ratings[event_id] = loaded.ranking_model((
            tf.constant([user_id], dtype=tf.int64),
            tf.constant([event_id], dtype=tf.int64)
            ))
    sorted_ratings = list(sorted(ratings.items(), key=lambda x: x[1][0][0], reverse=True))

    output = {}
    i = 0
    for event_id, score in sorted_ratings:
        event_name = list(events_df[events_df['id']==event_id]['event_name'])[0]
        output[i] = {
            "event_id": int(event_id),
            "event_name": event_name,
            "rating_prediction_score": float(score[0][0])
        }
        i += 1
    return output

def tags_search_model(query, top_n=1):
    events_df = df_loader.get_events_df()
    
    # to remove string quotes correctly, we need to parse data into list first, then join it back into string
    data = list(map(lambda x: ast.literal_eval(x), events_df['tags']))
    data = list(map(lambda x: ', '.join(x), data))

    vectorizer = tf.keras.layers.experimental.preprocessing.TextVectorization(
        max_tokens=None,
        output_mode='tf_idf',
        output_sequence_length=None,
        pad_to_max_tokens=False,
        standardize='lower_and_strip_punctuation',
        split='whitespace',
    )

    new_data = data + [query]
    vectorizer.adapt(new_data)
    X = vectorizer(new_data)

    cosine_sim = tf.linalg.matmul(X, X, transpose_b=True)
    sim_scores = list(enumerate(cosine_sim[-1, :-1]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_docs = sim_scores[:top_n]

    # handle multiple event with same list of tags
    output = {}
    c=0
    for index, score in top_docs:
        tags = new_data[index]
        tags_str = str(tags.split(', '))
        
        events_with_tags_df = events_df[events_df['tags']==tags_str]
        ids = list(events_with_tags_df['id'])
        event_names = list(events_with_tags_df['event_name'])
        for index in range(len(events_with_tags_df)):
            if c == top_n:
                break
            output[c] = {
                'event_id': ids[index],
                'event_name': event_names[index],
                'tags': tags,
                'match_score': float(score)
            }
            c += 1
    return output