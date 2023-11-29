import os
import requests
import wandb
import subprocess
from dotenv import load_dotenv

load_dotenv()

def fetch_data():
    url = 'https://dsserver-prod-resources-1.s3.amazonaws.com/nlp/train.csv'
    response = requests.get(url)

    with open('train.csv', 'wb') as file:
        file.write(response.content)


    chave_api = os.environ.get('KEY')
    wandb.login(key=chave_api)

    PROJECT_NAME = os.environ.get('PROJECT_NAME')
    comando = f'wandb artifact put --name {PROJECT_NAME}/train --type dataset --description "Positive and Negative Sentiment Analysis Dataset" train.csv'

    subprocess.run(comando, shell=True)
