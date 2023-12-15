import mlflow
import os
from dotenv import load_dotenv

load_dotenv()


def download_artifacts_by_run_name(run_name):
    EXPERIMENT_NAME = os.environ.get("EXPERIMENT_NAME")
    runs = mlflow.search_runs(experiment_ids=mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id,
                              filter_string=f"attributes.run_name='{run_name}'",
                              order_by=["start_time desc"],
                              max_results=1)
    
    if not runs.empty:
        run_id = runs.iloc[0]["run_id"]
        mlflow.artifacts.download_artifacts(run_id=run_id, dst_path=os.getcwd())
        print(f"Arquivo baixado com sucesso")
    else:
        print("Nenhum run encontrado.")
