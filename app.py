from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import joblib


app = FastAPI()

# Add CORS middleware if you are accessing the API from a frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the input data model for prediction
class InputData(BaseModel):
    Station_Region: int
    Temps_Min: float
    Temps_Max: float
    Rain: float
    AM9_RH: float
    AM9_Spd: float
    PM3_RH: float


@app.get("/")
async def root():
    return {"message": "Welcome to the FastAPI ML Model API!"}


@app.post("/predict")
async def predict(data: InputData):
    input_data = np.array([[data.Station_Region, data.Temps_Min, data.Temps_Max, data.Rain, data.AM9_RH, data.AM9_Spd, data.PM3_RH]])
    input_data = np.array(input_data).reshape(1, -1)

    try:
        # Load the model at the point of prediction (move it here for testing)
        model = joblib.load('BB_svm_model.pkl')  # Ensure correct path

        # Get prediction
        prediction = model.predict(input_data)
        # Get probability estimates
        prediction_proba = model.predict_proba(input_data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    return {
        "prediction": prediction.tolist(),  # Convert numpy to list
        "probability": prediction_proba.tolist()  # Convert numpy to list
    }
