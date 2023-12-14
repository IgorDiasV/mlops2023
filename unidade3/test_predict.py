from classification_text import Classifytext
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def test_predict():

    classificador = Classifytext() 

    test_data = pd.read_csv("test_data.csv")
    x_test = test_data['text']
    y_test = test_data['category']

    y_pred = classificador.predict_list_input(x_test)

    labels = classificador.encoder.classes_


    ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred, labels=labels), display_labels = labels).plot()
    plt.grid(False)
    plt.show()