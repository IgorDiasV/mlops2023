import numpy as np
import tensorflow as tf
import joblib
from transformers import AutoTokenizer

class Classifytext():
    def __init__(self):
        self.model = self.load_model()
        self.encoder = self.load_enconder()
        self.tokenizer = self.load_tokenizer()

    def load_model(self):
        model = tf.keras.saving.load_model("modelo.keras")
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
        print("resultado: ", predicted_category_label)
        return predicted_category_label