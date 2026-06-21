import mlflow

client = mlflow.MlflowClient()

for run in client.search_runs(
    experiment_ids=["1"]
):
    print(run.info.run_id)




print("----")
print(
    client.search_registered_models()
)