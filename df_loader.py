import pandas as pd

events_path = './events.csv'
ratings_path = './ratings.csv'

def get_events_df():
    return pd.read_csv(events_path)
def get_ratings_df():
    return pd.read_csv(ratings_path)
def get_combined_df():
    return pd.merge(get_ratings_df(), get_events_df(), left_on='event_id', right_on='id')
def get_processed_df():
    processed_df = pd.merge(get_ratings_df(), get_events_df(), left_on='event_id', right_on='id')
    location_df = pd.get_dummies(processed_df['location'])
    location_df.rename(columns=lambda x: f'location_{x}'.lower().replace(' ', '_'), inplace=True)
    processed_df = processed_df.drop(['id','date','description','interested','location','tags'],axis=1)
    return pd.concat([processed_df, location_df],axis=1)