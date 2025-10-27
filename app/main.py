from fastapi import FastAPI
import mlflow.sklearn
import pandas as pd

model=mlflow.sklearn.load_model("models:/wine-quality-data/latest")

app=FastAPI()

@app.post("/predict")
def predict(features: dict):
    df = pd.DataFrame([features])
    pred = model.predict(df)[0]
    return {"quality": "good" if pred==1 else "not good"}