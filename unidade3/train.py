import pandas as pd
import tensorflow as tf
from transformers import AutoTokenizer
from transformers import TFAutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import joblib
import mlflow
import mlflow.sklearn
import os

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

        run_name = 'data_segregation_run'
        runs = mlflow.search_runs(experiment_ids=mlflow.get_experiment_by_name("text_classification").experiment_id,
                                    filter_string=f"attributes.run_name='{run_name}'",
                                    order_by=["start_time desc"],
                                    max_results=1)
        if not runs.empty:
            run_id = runs.iloc[0]["run_id"]
            mlflow.artifacts.download_artifacts(run_id=run_id, dst_path=os.getcwd())
            print(f"Arquivo baixado com sucesso")
        else:
            print("Nenhum run encontrado para a etapa 'data_segregation'.")

        df_train = load_data("train_data.csv")
        df_test = load_data("test_data.csv")

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

        test_dataset = tf.data.Dataset.from_tensor_slices((
            dict(test_encodings),
            tf.constant(y_test, dtype=tf.int32)
        ))

        train_dataset = train_dataset.batch(16)
        test_dataset = test_dataset.batch(16)

        model = TFAutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=5)

        optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy')]

        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        # model.fit(train_dataset, epochs=1, validation_data=train_dataset)

        # model.save_weights('pesos_rede.h5')

        # joblib.dump(encoder, 'enconder')

        # mlflow.sklearn.log_model(model, "model")
        mlflow.log_artifact('enconder')
        mlflow.log_artifact('pesos_rede.h5')
