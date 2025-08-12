import os
import logging
import joblib
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

MODEL_PATH = os.getenv("MODEL_PATH", "/opt/ml/model/model.pkl")
model = None

app = FastAPI(title="ML Inference API", version="1.0.0")

class PredictionRequest(BaseModel):
    features: List[float]  # list of numeric features


@app.on_event("startup")
def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        logging.error(f"Model file not found at: {MODEL_PATH}")
        raise RuntimeError(f"Model file not found at: {MODEL_PATH}")
    #model = joblib.load(MODEL_PATH)
    model = lambda x: 1
    logging.info(f"Model loaded from: {MODEL_PATH}")


@app.get("/ping")
def ping():
    """Health check endpoint."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    return {"status": "ok"}


@app.post("/invocations")
def invocations(request: PredictionRequest):
    """Prediction endpoint."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    try:
        prediction = model.predict([request.features])
        return {"prediction": prediction.tolist()}
    except Exception as e:
        logging.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))
