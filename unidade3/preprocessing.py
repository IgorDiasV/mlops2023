import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import mlflow
import pandas as pd
import os
from dotenv import load_dotenv
from utils import download_artifacts_by_run_name

load_dotenv()

def preprocessing():
    with mlflow.start_run(run_name='preprocessing_run'):

        def punctuations(inputs):
            return re.sub(r'[^a-zA-Z]', ' ', inputs)

        def tokenization(inputs):
            return word_tokenize(inputs)

        def stopwords_remove(inputs):
            return [k for k in inputs if k not in stop_words]

        def lemmatization(inputs):
            return [lemmatizer.lemmatize(word=kk, pos='v') for kk in inputs]

        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('omw-1.4')

        download_artifacts_by_run_name('fetch_data_run')
        PATH_DATASET = os.environ.get("PATH_DATASET")

        df = pd.read_csv(PATH_DATASET)
        df['text'] = df['text'].str.lower()
        df['text'] = df['text'].apply(punctuations)
        df['text_tokenized'] = df['text'].apply(tokenization)

        stop_words = set(stopwords.words('english'))
        stop_words.remove('not')

        df['text_stop'] = df['text_tokenized'].apply(stopwords_remove)

        lemmatizer = WordNetLemmatizer()

        df['text_lemmatized'] = df['text_stop'].apply(lemmatization)
        df['final'] = df['text_lemmatized'].str.join(' ')

        PATH_CLEAN_DATA = os.environ.get("PATH_CLEAN_DATA")
        df.to_csv(PATH_CLEAN_DATA)
        mlflow.log_artifact(PATH_CLEAN_DATA)
