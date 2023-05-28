from typing import Union

from fastapi import FastAPI
import model_handler

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

@app.get("/retrain")
def retrain():
    model_handler.retrain()
    return "Retraining finished"