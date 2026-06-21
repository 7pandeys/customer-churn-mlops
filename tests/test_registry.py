import mlflow

client = mlflow.MlflowClient()

experiment = client.get_experiment_by_name(
    "customer-churn"
)

runs = client.search_runs(
    experiment_ids=[
        experiment.experiment_id
    ]
)

latest_run = runs[0]

run_id = latest_run.info.run_id

model_uri = (
    f"runs:/{run_id}/model"
)

result = mlflow.register_model(
    model_uri=model_uri,
    name="customer-churn-model"
)

print(result)