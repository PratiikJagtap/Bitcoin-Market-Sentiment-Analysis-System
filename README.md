# Bitcoin Market Behavior Modeling

## Project Overview

This project analyzes the relationship between **market sentiment** and **trading activity** and builds a **time-series classification model** to predict market sentiment (Fear, Neutral, Greed) using aggregated trading data.

The focus of the project is not price prediction, but **understanding trader behavior under different sentiment conditions** and deploying a complete end-to-end ML system.

---

## Problem Statement

Trading behavior changes based on market sentiment.  
The goal is to answer:

**Can daily trading activity be used to classify market sentiment reliably?**

This is solved as a **time-series classification problem** with proper handling of time order to avoid data leakage.

---

## Data Description

### 1. Sentiment Data (`fear_greed.csv`)
- One record per day
- Fear & Greed index value
- Sentiment category (target variable)

### 2. Trading Data (`Historical_data.csv`)
- Trade-level data (multiple trades per day)
- Price, volume, fees, PnL, direction
- Timestamp-based records

---

## Key Challenge

- Sentiment data is **daily**
- Trading data is **trade-level**

### Solution
Trading data is **aggregated to daily level**, then joined with sentiment data using the **date** column.

---

## Approach

### 1. Data Preprocessing
- Converted timestamps to datetime
- Extracted date from timestamps
- Removed ID and non-informative columns
- Encoded categorical values (Buy/Sell, flags)

### 2. Feature Engineering
Daily aggregation of trading data:
- Average price
- Total volume
- Number of trades
- Total fees
- Total PnL
- Buy/Sell activity metrics

### 3. Dataset Integration
- Joined trading and sentiment datasets on date
- Final dataset contains one row per day
- Date used only for ordering, not as a feature

---

## Exploratory Data Analysis (EDA)

Key observations:
- Trading activity patterns vary across sentiment classes
- Greed periods show higher volume and activity
- Fear periods show cautious trading behavior
- Some features are correlated but still informative
- Time-based trends confirm need for time-aware splitting

---

## Modeling

### Baseline Model
A simple baseline classifier was used to establish a reference performance.

### Models Evaluated
- Logistic Regression
- Random Forest
- Artificial Neural Network (ANN)

### Final Model Selection
**Logistic Regression** was selected because:
- Stable performance
- Better generalization
- No strong overfitting
- Well-suited for small, tabular datasets

Random Forest showed very high accuracy but signs of overfitting.  
ANN did not perform well, which is expected for this dataset type.

---

## Evaluation

- Time-based train–test split used
- Evaluated using classification metrics
- Focused on generalization rather than raw accuracy

---

## Model Packaging

The entire pipeline was saved as a single artifact:
- Feature scaler
- Trained Logistic Regression model

This ensures:
- Consistent preprocessing during inference
- No training–inference mismatch
- Easy deployment

---

## API Design (FastAPI)

A lightweight REST API was built using FastAPI.

### Endpoints

#### `/health`
Health check endpoint to verify service status.

#### `/predict`
Accepts raw market features and returns:
- Predicted sentiment class ID
- Sentiment label (Fear / Neutral / Greed)

Pydantic is used for input validation to ensure reliable inference.

---

## Local Testing

Before deployment:
- API endpoints tested locally
- Predictions verified across different market conditions
- API outputs matched notebook predictions
- Feature ordering and preprocessing validated

---

## Dockerization

- Dockerfile created to containerize the application
- Ensures consistent runtime environment
- Used mainly for local testing and deployment readiness

---

## Deployment

- Repository connected to Render via GitHub
- Automatic build and deployment on code push
- No manual Docker image handling required

---

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- FastAPI
- Pydantic
- Docker
- Render (Deployment)

---

## How to Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the API
uvicorn app:app --reload

