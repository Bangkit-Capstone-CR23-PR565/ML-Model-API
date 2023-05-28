import pandas as pd

events_path = './events.csv'
ratings_path = './ratings.csv'

def get_events_df():
    return pd.read_csv(events_path)
def get_ratings_df():
    return pd.read_csv(ratings_path)
def get_combined_df():
    return pd.merge(get_ratings_df(), get_events_df(), left_on='event_id', right_on='id')