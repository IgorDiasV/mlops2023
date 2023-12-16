import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from utils import download_artifacts_by_run_name
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

        download_artifacts_by_run_name('preprocessing_run')

        PATH_CLEAN_DATA = os.environ.get('PATH_CLEAN_DATA')
        PATH_TRAIN_DATA = os.environ.get('PATH_TRAIN_DATA')
        PATH_TEST_DATA = os.environ.get('PATH_TEST_DATA')
        
        data = load_data(PATH_CLEAN_DATA)
        x_train, x_test, y_train, y_test = split_data(data)
        train_data = pd.DataFrame({'text': x_train, 'category': y_train})
        test_data = pd.DataFrame({'text': x_test, 'category': y_test})
        train_data.to_csv(PATH_TRAIN_DATA, index=False)
        test_data.to_csv(PATH_TEST_DATA, index=False)

        mlflow.log_artifact(PATH_TRAIN_DATA)
        mlflow.log_artifact(PATH_TEST_DATA)