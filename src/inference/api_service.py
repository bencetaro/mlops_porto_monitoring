import os
import time
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from prometheus_client import Counter, Histogram, generate_latest

from src.inference.schemas import PredictionResponse, Item, BatchRequest
from src.inference.helpers import inference_preprocessing

MODEL_PATH = os.getenv("MODEL_PATH", "/output/model.pkl")
PREP_PATH = os.getenv("PREP_PATH", "/data/processed/preprocessors.pkl")
app = FastAPI()

REQUEST_COUNTER = Counter("inference_requests_total", "Total inference requests")
ERROR_COUNTER = Counter("inference_errors_total", "Total inference errors")
INFERENCE_TIME = Histogram("inference_duration_seconds", "Inference latency")

MODEL_CACHE = {}
CACHE_LIMIT = 5

def cache_model(key, model):
    if len(MODEL_CACHE) >= CACHE_LIMIT:
        MODEL_CACHE.clear()
    MODEL_CACHE[key] = model

def load_model(model_name):
    if model_name in MODEL_CACHE:
        return MODEL_CACHE[model_name]

    path = f"/output/models/{model_name}.pkl"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Model not found")

    model = joblib.load(path)
    cache_model(model_name, model)
    return model

def get_default_model():
    key = "default"
    if key in MODEL_CACHE:
        return MODEL_CACHE[key]

    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=500, detail="Default model missing")

    model = joblib.load(MODEL_PATH)
    cache_model(key, model)
    return model

@app.get("/models")
def list_models():
    path = "/output/models"
    if not os.path.exists(path):
        return []
    return sorted([f.replace(".pkl", "") for f in os.listdir(path)])

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return generate_latest().decode("utf-8")

@app.post("/predict", response_model=PredictionResponse)
def predict(item: Item, model_name: str = Query("default")):
    model = get_default_model() if model_name == "default" else load_model(model_name)

    df = pd.DataFrame([item.root])
    X = inference_preprocessing(df, PREP_PATH)

    if hasattr(model, "predict_proba"):
        pred = model.predict_proba(X)[0][1]
    else:
        pred = model.predict(X)[0]

    return {"prediction": float(pred)}

@app.post("/predict/batch", response_model=List[PredictionResponse])
def predict_batch(items: BatchRequest, model_name: str = Query("default")):
    model = get_default_model() if model_name == "default" else load_model(model_name)

    df = pd.DataFrame(items.root)
    X = inference_preprocessing(df, PREP_PATH)

    if hasattr(model, "feature_names_in_"):
        X = X[model.feature_names_in_]
    else:
        X = X[model.booster_.feature_name()]

    if hasattr(model, "predict_proba"):
        preds = model.predict_proba(X)[:, 1]
    else:
        preds = model.predict(X)

    return [{"prediction": float(p)} for p in preds]
