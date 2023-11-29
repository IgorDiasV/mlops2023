import os
import wandb
from dotenv import load_dotenv
import pandas as pd
from sklearn.model_selection import train_test_split
import datasets
import transformers
import tensorflow as tf
import tensorflow_datasets as tfds
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import TFAutoModelForSequenceClassification

load_dotenv()


def  load_data(path):
  data = ''
  for file in os.listdir(path):
    data = pd.read_csv(os.path.join(path, file))
  return data


def split_x_y(data):
  x = data['text']
  y = data['label']
  return x, y


chave_api = os.environ.get('KEY')
wandb.login(key=chave_api)

PROJECT_NAME = os.environ.get('PROJECT_NAME') 
wandb.init(project=PROJECT_NAME, job_type="train")

train_data_artifact = wandb.use_artifact(f'{PROJECT_NAME}/train_data:v0', type='TrainData')
train_data_dir = train_data_artifact.download()

test_data_artifact = wandb.use_artifact(f'{PROJECT_NAME}/test_data:v0', type='TestData')
test_data_dir = test_data_artifact.download()

df_train = load_data(train_data_dir)
df_test = load_data(test_data_dir)

X_train, y_train = split_x_y(df_train)
X_test, y_test = split_x_y(df_test)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")

train_encodings = tokenizer(list(X_train), truncation=True, padding=True)
test_encodings = tokenizer(list(X_test), truncation=True, padding=True)

train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    tf.constant(y_train.values, dtype=tf.int32)
))

test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_encodings),
    tf.constant(y_test.values, dtype=tf.int32)
))

train_dataset = train_dataset.batch(16)
test_dataset = test_dataset.batch(16)

model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5)

loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy')]

model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
model.fit(train_dataset, epochs=10, validation_data=train_dataset,
          callbacks=[wandb.keras.WandbCallback(save_model=False,
                                                   compute_flops=True)])