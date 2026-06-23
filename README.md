# Customer Churn MLOps Pipeline

## Overview

This project implements an end-to-end Machine Learning Operations (MLOps) pipeline for predicting customer churn using the Telco Customer Churn dataset.

The project covers the complete machine learning lifecycle including data ingestion, preprocessing, model training, evaluation, experiment tracking, model registry, API deployment, containerization, CI/CD, and monitoring.

---

## Architecture

Data Ingestion
→ Data Preprocessing
→ Feature Engineering
→ Model Training
→ Model Evaluation
→ MLflow Tracking
→ Model Registry
→ FastAPI Inference API
→ Docker Deployment
→ GitHub Actions CI/CD
→ Monitoring

---

## Tech Stack

### Machine Learning

* Python
* Pandas
* Scikit-Learn
* Logistic Regression

### MLOps

* MLflow
* Model Registry
* Joblib

### API

* FastAPI
* Swagger UI
* Pydantic

### DevOps

* Docker
* GitHub Actions
* Poetry

---

## Features

### Data Processing

* Data ingestion from CSV
* Missing value handling
* Feature encoding
* Feature scaling using StandardScaler

### Model Training

* Logistic Regression model
* Train/Test split
* Performance evaluation

### Experiment Tracking

* MLflow experiments
* Parameter logging
* Metric logging
* Artifact logging

### Model Registry

* Model versioning
* Model registration in MLflow Registry

### API Deployment

* REST API using FastAPI
* Swagger documentation
* Prediction endpoint

### Monitoring

* Health endpoint
* Metrics endpoint
* Request count monitoring
* Prediction latency monitoring

### CI/CD

* Automated testing
* Docker image build
* GitHub Actions workflow

---

## Project Structure

```text
customer-churn-mlops/

├── data/
│   └── churn.csv
│
├── models/
│   ├── churn_model.pkl
│   ├── churn_pipeline.pkl
│   └── features.pkl
│
├── src/
│   ├── ingest.py
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│   ├── evaluate.py
│   └── api.py
│
├── scripts/
│   └── train_and_log.py
│
├── tests/
│   ├── test_train.py
│   ├── test_predict.py
│   └── test_registry.py
│
├── Dockerfile
├── .github/
│   └── workflows/
│       └── ci.yml
│
└── README.md
```

---

## Model Performance

| Metric   | Value |
| -------- | ----- |
| Accuracy | ~0.81 |

---

## Running Locally

### Install Dependencies

```bash
poetry install
```

### Train Model

```bash
poetry run python scripts/train_and_log.py
```

### Run API

```bash
poetry run uvicorn src.api:app --reload
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

---

## API Endpoints

### Health Check

```http
GET /health
```

Response:

```json
{
  "status": "healthy"
}
```

---

### Metrics

```http
GET /metrics
```

Response:

```json
{
  "requests": 10
}
```

---

### Predict

```http
POST /predict
```

Example Request:

```json
{
  "gender": "Male",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 75.5,
  "TotalCharges": 906
}
```

Response:

```json
{
  "prediction": 1,
  "latency_ms": 12.5,
  "request_count": 3
}
```

---

## Future Improvements

* Prometheus Integration
* Grafana Dashboards
* Data Drift Detection
* Model Drift Detection
* Kubernetes Deployment
* AWS Deployment
* Automated Retraining Pipeline

---

## Key Learnings

* Machine Learning Pipeline Design
* Feature Engineering
* Model Evaluation
* MLflow Tracking
* Model Registry
* FastAPI Deployment
* Docker Containerization
* GitHub Actions CI/CD
* Monitoring and Observability
* Production ML Systems

---

## Author

Sandeep Pandey

Data Engineer | MLOps Engineer | AI Engineer Aspirant
