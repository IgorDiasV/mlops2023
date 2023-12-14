import numpy as np
import tensorflow as tf
import joblib
from transformers import AutoTokenizer
from transformers import TFAutoModelForSequenceClassification
class Classifytext():
    def __init__(self):
        self.model = self.load_model()
        self.encoder = self.load_enconder()
        self.tokenizer = self.load_tokenizer()

    def load_model(self):

        model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=5)
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5)
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy')]
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        model.load_weights('pesos_rede.h5')
        
        return model 

    def load_enconder(self):
        encoder = joblib.load('enconder')
        return encoder
    
    def load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        return tokenizer

    def predict_input_user(self, texto):
        user_encoding = self.tokenizer(texto, truncation=True, padding=True, return_tensors="tf")
        user_encoding = {key: np.array(value) for key, value in user_encoding.items()}

        predictions = self.model.predict(user_encoding)

        predicted_category = tf.argmax(predictions.logits, axis=1).numpy()[0]

        predicted_category_label = self.encoder.inverse_transform([predicted_category])[0]
 
        return predicted_category_label
    
    def predict_list_input(self, list_inputs):

        list_predict = []
        for texto in list_inputs:
            user_encoding = self.tokenizer(texto, truncation=True, padding=True, return_tensors="tf")
            user_encoding = {key: np.array(value) for key, value in user_encoding.items()}

            predictions = self.model.predict(user_encoding)

            predicted_category = tf.argmax(predictions.logits, axis=1).numpy()[0]

            predicted_category_label = self.encoder.inverse_transform([predicted_category])[0]
            list_predict.append(predicted_category_label)

        return list_predict