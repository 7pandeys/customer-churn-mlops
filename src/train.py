from sklearn.linear_model import LogisticRegression
import joblib

def train_model(
    X_train,
    y_train
):

    model = LogisticRegression(max_iter =1000)

    model.fit(
        X_train,
        y_train
    )
    # print(X_train.columns.tolist())
    # print(len(X_train.columns))
    joblib.dump(
        model,
        "models/churn_model.pkl"
    )

    joblib.dump(
        X_train.columns.tolist(),
        "models/features.pkl"
    )

    return model