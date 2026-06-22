from src.ingest import load_data
from src.preprocess import split_data
from src.train import train_model
from src.evaluate import evaluate

import mlflow
import mlflow.sklearn

# Load Data
df = load_data(
    "data/churn.csv"
)

# Split Data
X_train, X_test, y_train, y_test = split_data(
    df
)

# Experiment
mlflow.set_experiment(
    "customer-churn"
)

with mlflow.start_run():

    # Train
    model = train_model(
        X_train,
        y_train
    )

    # Evaluate
    score = evaluate(
        model,
        X_test,
        y_test
    )

    # Log Params
    mlflow.log_param(
        "model",
        "LogisticRegression"
    )

    mlflow.log_param(
        "max_iter",
        1000
    )

    mlflow.log_param(
        "features",
        len(X_train.columns)
    )

    # Log Metrics
    mlflow.log_metric(
        "accuracy",
        score
    )

    # Log Artifacts
    mlflow.log_artifact(
        "models/churn_model.pkl"
    )

    mlflow.log_artifact(
        "models/features.pkl"
    )

    # Log Model
    mlflow.sklearn.log_model(
        sk_model=model,
        name="model"
    )

    print(
        f"Accuracy: {score}"
    )