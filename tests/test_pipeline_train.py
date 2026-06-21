from src.ingest import load_data
from src.train_pipeline import pipeline

from sklearn.model_selection import train_test_split

df = load_data(
    "data/churn.csv"
)

df = df.drop(
    "customerID",
    axis=1
)

df["TotalCharges"] = (
    df["TotalCharges"]
    .replace(" ", 0)
    .astype(float)
)

X = df.drop(
    "Churn",
    axis=1
)

y = df["Churn"].map(
    {
        "Yes": 1,
        "No": 0
    }
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

pipeline.fit(
    X_train,
    y_train
)

score = pipeline.score(
    X_test,
    y_test
)

print(score)

import joblib

joblib.dump(
    pipeline,
    "models/churn_pipeline.pkl"
)