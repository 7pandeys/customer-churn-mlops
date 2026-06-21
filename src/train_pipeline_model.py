import joblib

pipeline.fit(
    X_train,
    y_train
)



joblib.dump(
    pipeline,
    "models/churn_pipeline.pkl"
)