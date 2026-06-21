from fastapi import FastAPI
from pydantic import BaseModel, Field
from src.ingest import load_data
from src.preprocess import split_data
from src.predict import load_model
from src.predict import load_pipeline

pipeline = load_pipeline()
df = load_data(
    "data/churn.csv"
)

X_train, X_test, y_train, y_test = split_data(
    df
)

model = load_model()

from pydantic import BaseModel


class Customer(BaseModel):

    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int

    PhoneService: str
    MultipleLines: str

    InternetService: str

    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str

    StreamingTV: str
    StreamingMovies: str

    Contract: str
    PaperlessBilling: str
    PaymentMethod: str

    MonthlyCharges: float
    TotalCharges: float

app = FastAPI()

@app.get("/")
def root():
    return {
        "message": "Customer Churn API"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


import pandas as pd


@app.post("/predict")
def predict(customer: Customer):

    df = pd.DataFrame(
        [customer.model_dump()]
    )

    prediction = pipeline.predict(
        df
    )

    return {
        "prediction":
        int(prediction[0])
    }

@app.get("/predict-test")
def predict_test():

    sample = X_test.iloc[[0]]

    prediction = model.predict(
        sample
    )

    return {
        "prediction":
        int(prediction[0])
    }