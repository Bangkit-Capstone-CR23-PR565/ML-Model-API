from typing import Union

from fastapi import FastAPI
from fastapi_utils.tasks import repeat_every
import model_handler
import model_trainer
import df_loader
import scraper

app = FastAPI()

@app.get("/")
async def index():
    return df_loader.get_processed_df().to_dict()

@app.get("/event/most-relevant/{user_id}")
async def items(user_id: int, limit: Union[int, None] = None):
    return model_handler.retrieval_model(user_id)[:limit]

@app.get("/event/top-recommendations/{user_id}")
async def rank(user_id: int, limit: Union[int, None] = None):
    return model_handler.ranking_model(user_id)[:limit]

@app.get("/event/search/{query}")
async def search(query: str, limit: Union[int, None] = None):
    return model_handler.tags_search_model(query, top_n=limit)

# Current: events from database, ratings from csv
@app.get("/update-data")
async def update_data():
    scraper.fetch_events()
    scraper.fetch_ratings()
    return "Data updated"

@app.get("/retrain")
async def retrain():
    model_trainer.retrain_all()
    return "Model retrained"

@app.on_event('startup')
@repeat_every(seconds=60*60*12, wait_first=True)   # schedule task every half day
async def scheduled_task():
    print(update_data())
    print(model_trainer.retrain_all())

# def create_db():
#     Event.__table__.create(engine)
#     return "created"

# def drop_db():
#     Event.__table__.create(engine)
#     return "dropped"