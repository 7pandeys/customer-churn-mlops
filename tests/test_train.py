from src.ingest import load_data
from src.preprocess import split_data
from src.train import train_model
from src.evaluate import evaluate

df = load_data(
    "data/churn.csv"
)

X_train, X_test, y_train, y_test = split_data(
    df
)

model = train_model(
    X_train,
    y_train
)

score = evaluate(
    model,
    X_test,
    y_test
)

print(score)