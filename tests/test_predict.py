# tests/test_predict.py

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

sample = X_test.iloc[[0]]

prediction = model.predict(
    sample
)

print(prediction)
print(y_test.iloc[0])