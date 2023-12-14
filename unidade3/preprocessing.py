import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import mlflow
import pandas as pd

def preprocessing():
    with mlflow.start_run(run_id='preprocessing', run_name='preprocessing_run'):

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

        df = pd.read_csv("bbc-text.csv")
        df['text'] = df['text'].str.lower()
        df['text'] = df['text'].apply(punctuations)
        df['text_tokenized'] = df['text'].apply(tokenization)

        stop_words = set(stopwords.words('english'))
        stop_words.remove('not')

        df['text_stop'] = df['text_tokenized'].apply(stopwords_remove)

        lemmatizer = WordNetLemmatizer()

        df['text_lemmatized'] = df['text_stop'].apply(lemmatization)
        df['final'] = df['text_lemmatized'].str.join(' ')

        path_clean_data = "clean_data.csv"
        df.to_csv(path_clean_data)
        mlflow.log_artifact(path_clean_data)
