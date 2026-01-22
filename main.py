import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# 1. Initialize App
app = FastAPI(title="Breast Cancer Detection API", version="1.0")

# 2. Load Model (Global Variable)
try:
    model = joblib.load('breast_cancer_model.pkl')
    print("Model loaded successfully.")
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

# 3. Define Input Data Structure (Updated for Pydantic V2)
class PatientData(BaseModel):
    features: List[float]

    class Config:
        json_schema_extra = {
            "example": {
                "features": [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 
                             0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 
                             0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 
                             25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 
                             0.2654, 0.4601, 0.1189]
            }
        }

# 4. API Endpoints
@app.get("/")
def home():
    return {"message": "ML Model is Live. Go to /docs to test it."}

@app.post("/predict")
def predict(data: PatientData):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Reshape input for sklearn (1 sample, 30 features)
    input_data = np.array(data.features).reshape(1, -1)

    if input_data.shape[1] != 30:
         raise HTTPException(status_code=400, detail=f"Expected 30 features, got {input_data.shape[1]}")

    # Predict
    prediction = model.predict(input_data)
    
    # 0 = Malignant, 1 = Benign
    class_name = "Benign" if prediction[0] == 1 else "Malignant"
    
    return {
        "diagnosis": class_name,
        "class_id": int(prediction[0])
    }
