from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Event
import os

load_dotenv()

engine = create_engine(f'mysql+pymysql://{os.getenv("DB_USER")}:{os.getenv("DB_PASS")}@{os.getenv("DB_HOST")}:{os.getenv("DB_PORT")}/{os.getenv("DB_NAME")}')
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

events_path = './events.csv'
ratings_path = './ratings.csv'

def get_events_df():
    db = SessionLocal()
    query = db.query(Event).all()
    db.close()
    df = pd.DataFrame([i.serialize for i in query])
    return df
    # return pd.read_csv(events_path)
def get_ratings_df():
    return pd.read_csv(ratings_path)
def get_combined_df():
    return pd.merge(get_ratings_df(), get_events_df(), left_on='event_id', right_on='id')
def get_processed_df():
    processed_df = pd.merge(get_ratings_df(), get_events_df(), left_on='event_id', right_on='id')
    processed_df = processed_df[['user_id','user_rating','event_id','location','category']]
    location_df = pd.get_dummies(processed_df['location'])
    location_df.rename(columns=lambda x: f'location_{x}'.lower().replace(' ', '_'), inplace=True)
    processed_df = processed_df.drop(['location'],axis=1)
    return pd.concat([processed_df, location_df],axis=1)