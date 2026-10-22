"""ClaimWise API: serves predictions from the trained risk model."""

from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from db import get_recent_predictions, init_db, save_prediction

MODEL_FILE = Path(__file__).parent / "models" / "risk_model.joblib"
model = joblib.load(MODEL_FILE)
init_db()

app = FastAPI(
    title="ClaimWise API",
    description="Predicts whether an insurance customer is likely to be high-cost.",
)


class Customer(BaseModel):
    """The customer details the API accepts, with validation rules."""

    age: int = Field(ge=18, le=100)
    sex: Literal["female", "male"]
    bmi: float = Field(gt=10, lt=70)
    children: int = Field(ge=0, le=10)
    smoker: Literal["yes", "no"]
    region: Literal["northeast", "northwest", "southeast", "southwest"]


def to_features(customer):
    """Turn a customer into the same columns the model was trained on."""
    row = {
        "age": customer.age,
        "bmi": customer.bmi,
        "children": customer.children,
        "sex_male": customer.sex == "male",
        "smoker_yes": customer.smoker == "yes",
        "region_northwest": customer.region == "northwest",
        "region_southeast": customer.region == "southeast",
        "region_southwest": customer.region == "southwest",
    }
    return pd.DataFrame([row])


@app.get("/health")
def health():
    """Simple check that the API is running."""
    return {"status": "ok"}


@app.post("/predict")
def predict(customer: Customer):
    """Predict whether a customer is likely to be high-cost."""
    features = to_features(customer)
    probability = round(float(model.predict_proba(features)[0][1]), 3)
    high_cost = probability >= 0.5
    prediction_id = save_prediction(customer, probability, high_cost)
    return {
        "prediction_id": prediction_id,
        "high_cost": high_cost,
        "probability": probability,
    }


@app.get("/predictions")
def recent_predictions(limit: int = 10):
    """Return the most recent predictions from the database."""
    return get_recent_predictions(limit)
