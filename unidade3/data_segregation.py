import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
import os

def load_data(data_path):
    data = pd.read_csv(data_path)
    return data

def split_data(data):
    X = data['final']
    y = data['category']
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=100)
    return x_train, x_test, y_train, y_test

def data_segregation():
    with mlflow.start_run(run_name="data_segregation_run"):

        run_name = 'preprocessing_run'
        runs = mlflow.search_runs(experiment_ids=mlflow.get_experiment_by_name("text_classification").experiment_id,
                                    filter_string=f"attributes.run_name='{run_name}'",
                                    order_by=["start_time desc"],
                                    max_results=1)
        if not runs.empty:
            run_id = runs.iloc[0]["run_id"]
            mlflow.artifacts.download_artifacts(run_id=run_id, dst_path=os.getcwd())
            print(f"Arquivo baixado com sucesso")
        else:
            print("Nenhum run encontrado para a etapa 'preprocessing'.")

        data_path = "clean_data.csv"
        data = load_data(data_path)
        x_train, x_test, y_train, y_test = split_data(data)
        train_data = pd.DataFrame({'text': x_train, 'category': y_train})
        test_data = pd.DataFrame({'text': x_test, 'category': y_test})
        train_data.to_csv('train_data.csv', index=False)
        test_data.to_csv('test_data.csv', index=False)

        mlflow.log_artifact('train_data.csv')
        mlflow.log_artifact('test_data.csv')