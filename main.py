from typing import Union

from fastapi import FastAPI, BackgroundTasks
from fastapi_utils.tasks import repeat_every
import model_handler
import model_trainer
import scraper

app = FastAPI()

@app.get("/")
def index():
    return {"Hello": "World"}

@app.get("/most-relevant/{user_id}")
def items(user_id: int):
    return model_handler.retrieval_model(user_id)

@app.get("/top-recommendations/{user_id}")
def rank(user_id: int):
   return model_handler.ranking_model(user_id)

@app.get("/search/{query}")
def search(query: str, top_n: Union[int, None] = 1):
    if top_n:
        return model_handler.tags_search_model(query, top_n=top_n)
    return model_handler.tags_search_model(query)

@app.get("/update-data")
def update_data():
    scraper.fetch_events()
    scraper.fetch_ratings()
    return "Data updated"

@app.get("/retrain")
async def retrain(background_tasks: BackgroundTasks):
    background_tasks.add_task(model_trainer.retrain_all())
    return "Retraining model..."

@app.on_event('startup')
@repeat_every(seconds=60*60*12, wait_first=True)   # schedule task every half day
def scheduled_task():
    print(update_data())
    print(model_trainer.retrain_all())
