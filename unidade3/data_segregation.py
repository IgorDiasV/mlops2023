import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split

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
    with mlflow.start_run():
        data_path = "clean_data.csv"
        data = load_data(data_path)
        x_train, x_test, y_train, y_test = split_data(data)
        train_data = pd.DataFrame({'text': x_train, 'category': y_train})
        test_data = pd.DataFrame({'text': x_test, 'category': y_test})
        train_data.to_csv('train_data.csv', index=False)
        test_data.to_csv('test_data.csv', index=False)

        mlflow.log_artifact('train_data.csv')
        mlflow.log_artifact('test_data.csv')