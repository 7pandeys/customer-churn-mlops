import joblib


def load_model():

    return joblib.load(
        "models/churn_model.pkl"
    )

def predict(
    model,
    features
):

    prediction = model.predict(
        features
    )

    return prediction[0]


def load_features():

    return joblib.load(
        "models/features.pkl"
    )

import joblib


def load_pipeline():

    return joblib.load(
        "models/churn_pipeline.pkl"
    )