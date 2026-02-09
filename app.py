from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI(title="Bitcoin Market Analysis API")

# Load trained model
with open("logistic_reg_.pkl", "rb") as f:
    model = pickle.load(f)


# Input schema
class MarketInput(BaseModel):
    avg_price: float
    total_volume: float
    trade_count: int
    total_fee: float
    avg_position: float
    avg_pnl: float
    side_mean: float
    direction_mean: float
    crossed_mean: float
    leverage_segment: int
    value: float   # Fear & Greed index value


APP_HTML = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>Vertex AI | Bitcoin Market Sentiment</title>
        <style>
        :root {
            color-scheme: light dark;
            --bg: #0f172a;
            --card: #111827;
            --accent: #38bdf8;
            --muted: #94a3b8;
            --text: #e2e8f0;
            --border: rgba(148, 163, 184, 0.35);
            --input: #0b1120;
            --input-border: rgba(148, 163, 184, 0.45);
        }
        body {
            margin: 0;
            font-family: "Inter", "Segoe UI", system-ui, sans-serif;
            background: radial-gradient(circle at top, #1e293b, #0f172a 45%);
            color: var(--text);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 32px 20px;
        }
        .container {
            max-width: 900px;
            width: 100%;
            display: grid;
            place-items: center;
        }
        .card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 28px;
            box-shadow: 0 20px 45px rgba(15, 23, 42, 0.35);
        }
        .card-header h1 {
            font-size: 2rem;
            margin: 0 0 6px;
        }
        .card-header p {
            margin: 0 0 20px;
            color: var(--muted);
            line-height: 1.6;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 16px;
        }
        label {
            font-size: 0.85rem;
            color: var(--muted);
            display: block;
            margin-bottom: 6px;
        }
        input {
            width: 100%;
            padding: 10px 5px;
            border-radius: 10px;
            border: 1px solid var(--input-border);
            background: var(--input);
            color: var(--text);
        }
        input:focus {
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.3);
        }
        button {
            margin-top: 18px;
            width: 100%;
            padding: 12px 16px;
            border-radius: 12px;
            border: none;
            background: var(--accent);
            color: #0b1220;
            font-weight: 700;
            cursor: pointer;
        }
        .result {
            margin-top: 18px;
            padding: 14px;
            border-radius: 12px;
            background: rgba(56, 189, 248, 0.12);
            border: 1px solid rgba(56, 189, 248, 0.4);
            font-weight: 600;
        }
        .meta {
            margin-top: 12px;
            font-size: 0.85rem;
            color: var(--muted);
        }
        </style>
    </head>
    <body>
        <div class="container">
        <section class="card">
            <div class="card-header">
            <h1>Bitcoin Sentiment Analysis</h1>
            <p>
                Fill in the market signals to get a predicted sentiment label and
                class ID from the model.
            </p>
            </div>
            <h2>Live Prediction</h2>
            <form id="prediction-form">
            <div class="grid">
                <div>
                <label>Average Price</label>
                <input name="avg_price" type="number" step="0.0001" required />
                </div>
                <div>
                <label>Total Volume</label>
                <input name="total_volume" type="number" step="0.0001" required />
                </div>
                <div>
                <label>Trade Count</label>
                <input name="trade_count" type="number" step="1" required />
                </div>
                <div>
                <label>Total Fee</label>
                <input name="total_fee" type="number" step="0.0001" required />
                </div>
                <div>
                <label>Average Position</label>
                <input name="avg_position" type="number" step="0.0001" required />
                </div>
                <div>
                <label>Average PnL</label>
                <input name="avg_pnl" type="number" step="0.0001" required />
                </div>
                <div>
                <label>Side Mean</label>
                <input name="side_mean" type="number" step="0.0001" required />
                </div>
                <div>
                <label>Direction Mean</label>
                <input name="direction_mean" type="number" step="0.0001" required />
                </div>
                <div>
                <label>Crossed Mean</label>
                <input name="crossed_mean" type="number" step="0.0001" required />
                </div>
                <div>
                <label>Leverage Segment</label>
                <input name="leverage_segment" type="number" step="1" required />
                </div>
                <div>
                <label>Fear & Greed Index</label>
                <input name="value" type="number" step="0.0001" required />
                </div>
            </div>
            <button type="submit">Generate Sentiment</button>
            </form>
            <div id="result" class="result" style="display: none"></div>
            <div id="meta" class="meta" style="display: none"></div>
        </section>
        </div>
        <script>
        const form = document.getElementById("prediction-form");
        const result = document.getElementById("result");
        const meta = document.getElementById("meta");

        form.addEventListener("submit", async (event) => {
            event.preventDefault();
            const formData = new FormData(form);
            const payload = Object.fromEntries(formData.entries());
            const parsed = {
            avg_price: Number(payload.avg_price),
            total_volume: Number(payload.total_volume),
            trade_count: Number(payload.trade_count),
            total_fee: Number(payload.total_fee),
            avg_position: Number(payload.avg_position),
            avg_pnl: Number(payload.avg_pnl),
            side_mean: Number(payload.side_mean),
            direction_mean: Number(payload.direction_mean),
            crossed_mean: Number(payload.crossed_mean),
            leverage_segment: Number(payload.leverage_segment),
            value: Number(payload.value),
            };

            const response = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(parsed),
            });

            if (!response.ok) {
            result.style.display = "block";
            meta.style.display = "none";
            result.textContent = "Prediction failed. Please verify inputs.";
            return;
            }

            const data = await response.json();
            result.style.display = "block";
            meta.style.display = "block";
            result.textContent = `Prediction: ${data.predicted_label}`;
            meta.textContent = `Class ID: ${data.predicted_class_id}`;
        });
        </script>
    </body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def ui_home():
    return APP_HTML


@app.get("/health")
def health_check():
    return {"status": "API running"}


@app.post("/predict")
def predict(data: MarketInput):
    features = np.array([[
    data.avg_price,
    data.total_volume,
    data.trade_count,
    data.total_fee,
    data.avg_position,
    data.avg_pnl,
    data.side_mean,
    data.direction_mean,
    data.crossed_mean,
    data.value,             
    data.leverage_segment   
]])


    prediction = model.predict(features)[0]

    CLASS_MAPPING = {
        0: "Extreme Fear",
        1: "Fear",
        2: "Neutral",
        3: "Greed",
        4: "Extreme Greed",
    }

    return {
        "predicted_class_id": int(prediction),
        "predicted_label": CLASS_MAPPING[int(prediction)]
    }