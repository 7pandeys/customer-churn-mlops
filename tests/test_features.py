from src.ingest import load_data
from src.preprocess import split_data

df = load_data(
    "data/churn.csv"
)

X_train, X_test, y_train, y_test = split_data(
    df
)

print(
    X_train.columns.tolist()
)