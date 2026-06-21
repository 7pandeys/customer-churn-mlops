from sklearn.metrics import (
    accuracy_score
)


def evaluate(
    model,
    X_test,
    y_test
):

    predictions = model.predict(
        X_test
    )

    score = accuracy_score(
        y_test,
        predictions
    )

    return score