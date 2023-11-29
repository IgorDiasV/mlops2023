import os
import wandb
from dotenv import load_dotenv
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

load_dotenv()


def preprocessing():
    def punctuations(inputs):
        return re.sub(r'[^a-zA-Z]', ' ', inputs)


    def tokenization(inputs):
        return word_tokenize(inputs)


    def stopwords_remove(inputs):
        return [k for k in inputs if k not in stop_words]


    def lemmatization(inputs):
        return [lemmatizer.lemmatize(word=kk, pos='v') for kk in inputs]


    chave_api = os.environ.get('KEY')
    wandb.login(key=chave_api)

    PROJECT_NAME = os.environ.get('PROJECT_NAME')
    wandb.init(project=f'{PROJECT_NAME}', save_code=True)
    artifact = wandb.use_artifact('train:v0')
    artifact_dir = artifact.download()

    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

    df = pd.read_csv(artifact_dir + '/train.csv')

    df = df.drop(['id', 'keyword', 'location'], axis=1)
    df['text'] = df['text'].str.lower()

    df['text'] = df['text'].apply(punctuations)

    df['text_tokenized'] = df['text'].apply(tokenization)

    stop_words = set(stopwords.words('english'))
    stop_words.remove('not')

    df['text_stop'] = df['text_tokenized'].apply(stopwords_remove)

    lemmatizer = WordNetLemmatizer()

    df['text_lemmatized'] = df['text_stop'].apply(lemmatization)
    df['final'] = df['text_lemmatized'].str.join(' ')

    clean_data_artifact = wandb.Artifact(
        name='clean_data',
        type='CleanData',
        description='Cleaned dataset'
    )
    path_clean_data = artifact_dir + "/clean_data.csv"
    df.to_csv(path_clean_data)
    clean_data_artifact.add_file(path_clean_data)
    wandb.log_artifact(clean_data_artifact)

    wandb.finish()
