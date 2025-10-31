

#expected input features for the model
#response format
#All types numeric since label encoding is applied during data prep

#Features from x-train, in order are: age, hypertension, heart_disease, marital_status, work_type, Residence_type, avg_glucose_level, bmi, smoking_status

#accepts record and returns probabilities and class preductions

from pydantic import BaseModel, Field
from typing import List, Union #import Union for possible missing values

class StrokeFeatures(BaseModel):
    #Field names MUST match columns in data/processed/X_train.csv exactly
    gender: int = Field(..., description="Label-encoded gender")
    age: float = Field(..., description="Age of the patient") #float
    hypertension: int = Field(..., description="0 if no hypertension, 1 if hypertension")
    heart_disease: int = Field(..., description="0 if no heart disease, 1 if heart disease")
    ever_married: int = Field(..., description="Label-encoded marital status")
    work_type: int = Field(..., description="Label-encoded work type") #0-4
    Residence_type: int = Field(..., description="Label-encoded residence type") #0 (Rural) or 1 (Urban)
    avg_glucose_level: float = Field(..., description="Average glucose level") #float
    bmi: Union[float, None] = Field(..., description="Body Mass Index, can be null") #float
    smoking_status: int = Field(..., description="Label-encoded smoking status") #0-3

class PredictionRequest(BaseModel):
    inputs: Union[StrokeFeatures, List[StrokeFeatures]]

class PredictionResult(BaseModel):
    prob: float = Field(..., description="Predicted probability of stroke (0-1)")
    label: int = Field(..., description="Predicted class 0/1")

class PredictionResponse(BaseModel):
    results: List[PredictionResult] 
    model_version: str