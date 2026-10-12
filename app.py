"""API REST FastAPI exposant la prédiction de churn."""

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from model_pipeline import load_model, predict

MODEL_PATH = os.getenv("MODEL_PATH", "classifier.joblib")

app = FastAPI(title="Churn Prediction API", version="1.0")

try:
    model = load_model(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")
except Exception as exc:
    model = None
    print(f"Erreur de chargement du modèle : {exc}")


class ChurnInput(BaseModel):
    """Caractéristiques d'un client."""

    CreditScore: float
    Gender: float
    Age: float
    Tenure: float
    Balance: float
    NumOfProducts: float
    HasCrCard: float
    IsActiveMember: float
    EstimatedSalary: float


@app.get("/")
def root():
    """Vérifie que l'API est en ligne."""
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict_route(data: ChurnInput):
    """Effectue une prédiction à partir des caractéristiques du client."""
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    try:
        return {"prediction": predict(model, data.model_dump())}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
