import mlflow
import requests

def fetch_data():
    with mlflow.start_run(run_name='fetch_data_run'):

        url = 'https://storage.googleapis.com/dataset-uploader/bbc/bbc-text.csv'
        response = requests.get(url)

        with open('bbc-text.csv', 'wb') as file:
            file.write(response.content)

        mlflow.log_artifact('bbc-text.csv')