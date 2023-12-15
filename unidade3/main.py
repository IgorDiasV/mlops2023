import mlflow
from pipeline import Pipeline
from fetch_data import fetch_data
from preprocessing import preprocessing
from data_segregation import data_segregation
from train import train
from test_predict import test_predict
from dotenv import load_dotenv
import os

load_dotenv()

URL_MLFLOW = os.environ.get("URL_MLFLOW")
EXPERIMENT_NAME = os.environ.get("EXPERIMENT_NAME")

mlflow.set_tracking_uri(URL_MLFLOW)
mlflow.set_experiment(experiment_name=EXPERIMENT_NAME)

pipeline = Pipeline()

@pipeline.task()
def run_fetch_data():
    print("running fetch data")
    fetch_data()


@pipeline.task(depends_on=run_fetch_data)
def run_preprocessing():
    print("running preprocessing")
    preprocessing()


@pipeline.task(depends_on=run_preprocessing)
def run_data_segregation():
    print("running data segregation")
    data_segregation()


@pipeline.task(depends_on=run_data_segregation)
def run_train():
    print("running train")
    train()

@pipeline.task(depends_on=run_train)
def run_test():
    print("running test")
    test_predict()

pipeline.run()
