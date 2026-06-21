from src.ingest import load_data
from src.preprocess import split_data
from src.train import train_model
from src.evaluate import evaluate

import mlflow

df = load_data(
    "data/churn.csv"
)

X_train, X_test, y_train, y_test = split_data(
    df
)

mlflow.set_experiment(
    "customer-churn"
)

with mlflow.start_run():

    model = train_model(
        X_train,
        y_train
    )

    score = evaluate(
        model,
        X_test,
        y_test
    )

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

    mlflow.log_metric(
        "accuracy",
        score
    )

    mlflow.log_artifact(
        "models/churn_model.pkl"
    )

    mlflow.log_artifact(
        "models/features.pkl"
    )

    print(score)