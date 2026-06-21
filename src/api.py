from fastapi import FastAPI
from pydantic import BaseModel, Field
from src.ingest import load_data
from src.preprocess import split_data
from src.predict import load_model

df = load_data(
    "data/churn.csv"
)

X_train, X_test, y_train, y_test = split_data(
    df
)

model = load_model()

class Customer(BaseModel):

    tenure: float = Field(example=10)

    MonthlyCharges: float = Field(
        example=50
    )

    TotalCharges: float = Field(
        example=500
    )

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


@app.post("/predict")
def predict_churn(
    customer: Customer
):

    return {
        "received": customer.model_dump()
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