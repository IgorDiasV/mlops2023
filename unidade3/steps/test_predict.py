from classification_text import Classifytext
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import os
from utils import download_artifacts_by_run_name
from dotenv import load_dotenv

load_dotenv()
def test_predict():

    download_artifacts_by_run_name("train_run")
   
    classificador = Classifytext() 

    PATH_TEST_DATA = os.environ.get("PATH_TEST_DATA")
    test_data = pd.read_csv(PATH_TEST_DATA)
    x_test = test_data['text']
    y_test = test_data['category']

    y_pred = classificador.predict_list_input(x_test)

    labels = classificador.encoder.classes_


    ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred, labels=labels), display_labels = labels).plot()

    plt.savefig('matriz_conf.png')
    mlflow.log_artifact("matriz_conf.png")
