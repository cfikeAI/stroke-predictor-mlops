

#expected input features for the model
#response format
#All types numeric since label encoding is applied during data prep

#Features from x-train, in order are: age, hypertension, heart_disease, marital_status, work_type, Residence_type, avg_glucose_level, bmi, smoking_status

#accepts record and returns probabilities and class preductions

from pydantic import BaseModel, Field
from typing import List, Literal, Union #import Union for possible missing values

# Maps human-readable values to numeric encodings from training
ENCODERS = {
    "gender": {"Male": 0, "Female": 1, "Other": 2},
    "ever_married": {"No": 0, "Yes": 1},
    "work_type": {
        "children": 0,
        "Govt_job": 1,
        "Never_worked": 2,
        "Private": 3,
        "Self-employed": 4
    },
    "Residence_type": {"Rural": 0, "Urban": 1},
    "smoking_status": {
        "never smoked": 0,
        "formerly smoked": 1,
        "smokes": 2,
        "Unknown": 3
    },
}

def encode_features(sample: dict) -> dict:
    """Convert human-readable string features to numeric encodings."""
    encoded = {}
    for key, value in sample.items():
        if key in ENCODERS:
            mapping = ENCODERS[key]
            # allow both string and numeric input
            if isinstance(value, str):
                encoded[key] = mapping.get(value, None)
            else:
                encoded[key] = value
        else:
            encoded[key] = value
    return encoded


class StrokeFeatures(BaseModel):
    #Field names MUST match columns in data/processed/X_train.csv exactly
    #gender: int = Field(..., description="Label-encoded gender")
    #age: float = Field(..., description="Age of the patient") #float
    #hypertension: int = Field(..., description="0 if no hypertension, 1 if hypertension")
    #heart_disease: int = Field(..., description="0 if no heart disease, 1 if heart disease")
    #ever_married: int = Field(..., description="Label-encoded marital status")
    #work_type: int = Field(..., description="Label-encoded work type") #0-4
    #Residence_type: int = Field(..., description="Label-encoded residence type") #0 (Rural) or 1 (Urban)
    #avg_glucose_level: float = Field(..., description="Average glucose level") #float
    #bmi: Union[float, None] = Field(..., description="Body Mass Index, can be null") #float
    #smoking_status: int = Field(..., description="Label-encoded smoking status") #0-3

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