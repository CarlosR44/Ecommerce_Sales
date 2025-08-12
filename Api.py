from fastapi import FastAPI, HTTPException
from preprocess import Preprocessor
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np

app = FastAPI()

class PredictInput(BaseModel):
    Product_Name: str = Field(..., alias="Product Name")
    Category: str
    Price: float
    Quantity: int
    Customer_Age: int = Field(..., alias="Customer Age")
    Customer_Gender: str = Field(..., alias="Customer Gender")
    Discount: float
    Payment_Method: str = Field(..., alias="Payment Method")

    class Config:
        validate_by_name = True


preprocessor = joblib.load("preprocessor.pkl")
target_scaler = joblib.load('target_scaler.pkl')
model = joblib.load('modelo_xgb.pkl')

@app.post("/predict")
def predict(data: PredictInput):
    df = pd.DataFrame([data.dict(by_alias=True)])
    df_preprocessed = preprocessor.transform(df)
    prediction_scaled = model.predict(df_preprocessed)
    prediction_original = target_scaler.inverse_transform(
        np.array(prediction_scaled).reshape(-1, 1)
    )
    return {"prediction": float(prediction_original[0, 0])}
