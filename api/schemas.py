

#expected input features for the model
#response format
#All types numeric since label encoding is applied during data prep

#Features from x-train, in order are: age, hypertension, heart_disease, marital_status, work_type, Residence_type, avg_glucose_level, bmi, smoking_status

#accepts record and returns probabilities and class preductions

from pydantic import BaseModel, Field
from typing import List, Literal, Union #import Union for possible missing values




class StrokeFeatures(BaseModel):

    gender: Literal["Male", "Female", "Other"] = Field(..., description="Patient gender")
    age: float = Field(..., gt=0, description="Age of the patient (years)")
    hypertension: Literal[0, 1] = Field(..., description="0 if no hypertension, 1 if hypertension")
    heart_disease: Literal[0, 1] = Field(..., description="0 if no heart disease, 1 if heart disease")
    ever_married: Literal["No", "Yes"] = Field(..., description="Marital status")
    work_type: Literal["children", "Govt_job", "Never_worked", "Private", "Self-employed"] = Field(..., description="Type of employment")
    Residence_type: Literal["Rural", "Urban"] = Field(..., description="Residential type")
    avg_glucose_level: float = Field(..., ge=0, description="Average glucose level in blood")
    bmi: Union[float, None] = Field(None, ge=0, description="Body Mass Index (can be null if missing)")
    smoking_status: Literal["formerly smoked", "never smoked", "smokes", "Unknown"] = Field(..., description="Smoking behavior")

class PredictionRequest(BaseModel):
    inputs: Union[StrokeFeatures, List[StrokeFeatures]]

class PredictionResult(BaseModel):
    prob: float = Field(..., description="Predicted probability of stroke (0-1)")
    label: int = Field(..., description="Predicted class 0/1")

class PredictionResponse(BaseModel):
    results: List[PredictionResult] 
    model_version: str