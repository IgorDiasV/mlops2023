import mlflow
import requests
from dotenv import load_dotenv
import os
load_dotenv()

def fetch_data():
    with mlflow.start_run(run_name='fetch_data_run'):

        url = 'https://storage.googleapis.com/dataset-uploader/bbc/bbc-text.csv'
        response = requests.get(url)
        PATH_DATASET = os.environ.get("PATH_DATASET")
        with open(PATH_DATASET, 'wb') as file:
            file.write(response.content)

        mlflow.log_artifact(PATH_DATASET)