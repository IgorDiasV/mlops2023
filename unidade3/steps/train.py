import pandas as pd
import tensorflow as tf
from transformers import AutoTokenizer
from transformers import TFAutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import joblib
import mlflow
import mlflow.sklearn
import os
from dotenv import load_dotenv
from utils import download_artifacts_by_run_name
load_dotenv()


def load_data(data_path):
    data = pd.read_csv(data_path)
    return data


def split_x_y(data):
    x = data['text']
    y = data['category']
    return x, y


def train():
    with mlflow.start_run(run_name='train_run'):
        mlflow.autolog()

        download_artifacts_by_run_name('data_segregation_run')
        
        PATH_TRAIN_DATA = os.environ.get("PATH_TRAIN_DATA")
        PATH_TEST_DATA = os.environ.get("PATH_TEST_DATA")
        
        df_train = load_data(PATH_TRAIN_DATA)
        df_test = load_data(PATH_TEST_DATA)

        X_train, y_train = split_x_y(df_train)
        X_test, y_test = split_x_y(df_test)

        encoder = LabelEncoder()
        encoder.fit(y_train)
        y_train = encoder.transform(y_train)
        y_test = encoder.transform(y_test)

        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

        train_encodings = tokenizer(
            list(X_train), truncation=True, padding=True)
        test_encodings = tokenizer(list(X_test), truncation=True, padding=True)

        train_dataset = tf.data.Dataset.from_tensor_slices((
            dict(train_encodings),
            tf.constant(y_train, dtype=tf.int32)
        ))

        train_dataset = train_dataset.batch(16)

        model = TFAutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=5)

        optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy')]

        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        
        PATH_WEIGHTS = os.environ.get("PATH_WEIGHTS")
        PATH_ENCONDER = os.environ.get("PATH_ENCONDER")
        model.fit(train_dataset, epochs=10, validation_data=train_dataset)

        model.save_weights(PATH_WEIGHTS)

        joblib.dump(encoder, PATH_ENCONDER)

        mlflow.sklearn.log_model(model, "model")
        mlflow.log_artifact(PATH_ENCONDER)
        mlflow.log_artifact(PATH_WEIGHTS)
