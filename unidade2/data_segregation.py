import os
import wandb
from dotenv import load_dotenv
import subprocess
import pandas as pd
from sklearn.model_selection import train_test_split

load_dotenv()

def load_data(data_path):
    data = pd.read_csv(os.path.join(data_path, 'clean_data.csv'))
    return data

def split_data(data):
    X = data['final']
    y = data['target']
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=100)

    return x_train, x_test, y_train, y_test


chave_api = os.environ.get('KEY')
wandb.login(key=chave_api)

PROJECT_NAME = os.environ.get('PROJECT_NAME')
run = wandb.init(project=PROJECT_NAME, job_type='data_segregation')
artifact = run.use_artifact('clean_data:latest')
data_path = artifact.download()

data = load_data(data_path)
x_train, x_test, y_train, y_test = split_data(data)

train_data = pd.DataFrame({'text': x_train, 'label': y_train})
test_data = pd.DataFrame({'text': x_test, 'label': y_test})

wandb.log({'train_data_shape': train_data.shape,
           'test_data_shape': test_data.shape})

train_data.to_csv('train_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)

train_artifact = wandb.Artifact(
    name='train_data',
    type='TrainData',
    description='Training data split from clean_data'
)
test_artifact = wandb.Artifact(
    name='test_data',
    type='TestData',
    description='Testing data split from clean_data'
)

train_artifact.add_file('train_data.csv')
test_artifact.add_file('test_data.csv')

run.log_artifact(train_artifact)
run.log_artifact(test_artifact)

wandb.finish()