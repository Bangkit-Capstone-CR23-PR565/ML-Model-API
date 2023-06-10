from typing import Union

from fastapi import FastAPI
from fastapi_utils.tasks import repeat_every
from starlette.responses import RedirectResponse
import model_handler
import model_trainer

app = FastAPI(
    title='ML-Model-API',
    description='This is a documentation for Capstone Project Event.mu API for serving machine learning model.',
)

@app.get("/",
         tags=['Events'],
         description='Redirects page to docs.')
async def get_index():
    return RedirectResponse(url='/docs')

@app.get("/events/most-relevant/{user_id}",
         tags=['Events'],
         description='Returns list of events sorted from most relevant of user with given id.')
async def get_most_relevant(user_id: int, limit: Union[int, None] = None):
    return model_handler.retrieval_model(user_id)[:limit]

@app.get("/events/top-ranking-predictions/{user_id}",
         tags=['Events'],
         description='Returns list of events sorted from highest score prediction of user with given id.')
async def get_top_ranking_predictions(user_id: int, limit: Union[int, None] = None):
    return model_handler.ranking_model(user_id)[:limit]

@app.get("/events/top-recommendations/{user_id}",
         tags=['Events'],
         description='Returns list of events sorted by most relevant, then picked out starting from the highest score prediction of user with given id.')
async def get_top_recommendations(user_id: int, limit: Union[int, None] = None):
    retrieval = model_handler.retrieval_model(user_id)
    limit = limit if limit else len(retrieval)
    retrieval_ids = set()
    for i in retrieval:
        if len(retrieval_ids) >= limit:
            break
        retrieval_ids.add(i['id'])
    ranking = model_handler.ranking_model(user_id)
    ranking_filtered = [i for i in ranking if i['id'] in retrieval_ids]
    return ranking_filtered

@app.get("/events/search/{query}",
         tags=['Events'],
         description='Returns list of events sorted from highest match score of given text query.')
async def search_events(query: str, limit: Union[int, None] = None):
    return model_handler.tags_search_model(query, top_n=limit)

@app.get("/retrain",
         tags=['Events'],
         description='Retrains and updates current model used.')
async def retrain_model():
    model_trainer.retrain_all()
    return "Model retrained"

@app.on_event('startup')
@repeat_every(seconds=60*60*12, wait_first=True)   # schedule task every half day
async def scheduled_task():
    print(model_trainer.retrain_all())