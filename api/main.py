"""
__author__ = "sato064"
"""

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from typing import Dict

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello Fast API"}

@app.post("/")
async def post(item: Dict):
    string = item["msg"]
    """
    Insert generating code here.
    """
    json_obj = jsonable_encoder(item)
    return json_obj

