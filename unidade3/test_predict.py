from classification_text import Classifytext
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import os

def test_predict():

    run_name = 'train_run'
    runs = mlflow.search_runs(experiment_ids=mlflow.get_experiment_by_name("text_classification").experiment_id,
                                filter_string=f"attributes.run_name='{run_name}'",
                                order_by=["start_time desc"],
                                max_results=1)
    
    run_id = runs.iloc[0]["run_id"]
    mlflow.artifacts.download_artifacts(run_id=run_id, dst_path=os.getcwd())
    classificador = Classifytext() 

    test_data = pd.read_csv("test_data.csv").head(5)
    x_test = test_data['text']
    y_test = test_data['category']

    y_pred = classificador.predict_list_input(x_test)

    labels = classificador.encoder.classes_


    ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred, labels=labels), display_labels = labels).plot()

    plt.savefig('matriz_conf.png')
    mlflow.log_artifact("matriz_conf.png")
