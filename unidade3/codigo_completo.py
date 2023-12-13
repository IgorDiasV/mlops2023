import os
import pandas as pd
import re
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import train_test_split
import requests
from sklearn.model_selection import train_test_split
import datasets
import transformers
import tensorflow as tf
# import tensorflow_datasets as tfds
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import TFAutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import joblib


def fetch_data():
  url = 'https://storage.googleapis.com/dataset-uploader/bbc/bbc-text.csv'
  response = requests.get(url)

  with open('bbc-text.csv', 'wb') as file:
      file.write(response.content)

def preprocessing():
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

    # df = df.drop(['id', 'keyword', 'location'], axis=1)
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

    data_path = "clean_data.csv"

    data = load_data(data_path)
    x_train, x_test, y_train, y_test = split_data(data)

    train_data = pd.DataFrame({'text': x_train, 'category': y_train})
    test_data = pd.DataFrame({'text': x_test, 'category': y_test})


    train_data.to_csv('train_data.csv', index=False)
    test_data.to_csv('test_data.csv', index=False)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


def split_x_y(data):
    x = data['text']
    y = data['category']
    return x, y


def train():
  df_train = load_data("train_data.csv")
  df_test = load_data("test_data.csv")

  X_train, y_train = split_x_y(df_train)
  X_test, y_test = split_x_y(df_test)

  encoder = LabelEncoder()
  encoder.fit(y_train)
  y_train = encoder.transform(y_train)
  y_test = encoder.transform(y_test)



  model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")

  train_encodings = tokenizer(list(X_train), truncation=True, padding=True)
  test_encodings = tokenizer(list(X_test), truncation=True, padding=True)

  train_dataset = tf.data.Dataset.from_tensor_slices((
      dict(train_encodings),
      tf.constant(y_train, dtype=tf.int32)
  ))

  test_dataset = tf.data.Dataset.from_tensor_slices((
      dict(test_encodings),
      tf.constant(y_test, dtype=tf.int32)
  ))

  train_dataset = train_dataset.batch(16)
  test_dataset = test_dataset.batch(16)

  model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=5)

  optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5)

  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy')]

  model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
  model.fit(train_dataset, epochs=10, validation_data=train_dataset)
  
  return model, encoder


fetch_data()
preprocessing()
data_segregation()
model, encoder = train()

model.save_weights('pesos_rede.h5')
joblib.dump(encoder, 'enconder')