import os
import wandb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

load_dotenv()

def eda():
    chave_api = os.environ.get('KEY')
    wandb.login(key=chave_api)

    PROJECT_NAME = os.environ.get('PROJECT_NAME')

    wandb.init(project=f'{PROJECT_NAME}', save_code=True)
    artifact = wandb.use_artifact('train:v0')
    artifact_dir = artifact.download()

    df = pd.read_csv(artifact_dir + '/train.csv')

    df = df.drop(['id','keyword', 'location'], axis=1)
    print(df['target'].value_counts(normalize=True))

    sns.countplot(data = df, x = 'target')
    plt.title('Tweet Count by Category')
    plt.savefig('file_size_distribution.png') 
    plt.close()

    wandb.log({"File Size Distribution": wandb.Image('file_size_distribution.png')})
    wandb.finish()