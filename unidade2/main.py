from pipeline import Pipeline
from fetch_data import fetch_data
from preprocessing import preprocessing
from data_check import data_check
from data_segregation import data_segregation
from train import train

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
def run_data_check():
    print("running data check")
    data_check()


@pipeline.task(depends_on=run_data_check)
def run_data_segregation():
    print("running data segregation")
    data_segregation()


@pipeline.task(depends_on=run_data_segregation)
def run_train():
    print("running train")
    train()


pipeline.run()
