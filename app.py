from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI(title="Bitcoin Market Analysis API")

# Load trained model
with open("random_forest_model.pkl", "rb") as f:
    model = pickle.load(f)


# Input schema
class MarketInput(BaseModel):
    avg_price: float
    total_volume: float
    total_usd: float
    trade_count: int
    total_fee: float
    avg_position: float
    avg_pnl: float
    side_mean: float
    direction_mean: float
    crossed_mean: float
    value: float   # Fear & Greed index value


@app.get("/")
def health_check():
    return {"status": "API running"}


@app.post("/predict")
def predict(data: MarketInput):
    features = np.array([[
        data.avg_price,
        data.total_volume,
        data.total_usd,
        data.trade_count,
        data.total_fee,
        data.avg_position,
        data.avg_pnl,
        data.side_mean,
        data.direction_mean,
        data.crossed_mean,
        data.value
    ]])

    prediction = model.predict(features)[0]

    return {
        "predicted_class": int(prediction)
    }
