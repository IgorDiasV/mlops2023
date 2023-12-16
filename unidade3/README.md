# Classifying topics with: Sport, Business, Technology, Politics and Entertainment

<!-- Shields Exemplo, existem N diferentes shield em https://shields.io/ -->
![GitHub last commit](https://img.shields.io/github/last-commit/IgorDiasV/mlops2023)
![GitHub language count](https://img.shields.io/github/languages/count/IgorDiasV/mlops2023)
![Github repo size](https://img.shields.io/github/repo-size/IgorDiasV/mlops2023)
![Github stars](https://img.shields.io/github/stars/IgorDiasV/mlops2023?style=social)

![Capa do Projeto](https://pbs.twimg.com/media/E7ktoEvX0AsJpXO.png)

> Case study: a public dataset from the BBC comprised of 2225 articles, each labeled under one of 5 categories: business, entertainment, politics, sport or tech. The dataset is broken into 1490 records for training and 735 for testing. The goal will be to build a system that can accurately classify previously unseen news articles into the right category.


### Group members:
- Igor Dias Verissimo Oliveira

- Matheus Dos Santos Lopes Rodrigues


## Prerequisites

Before you begin, make sure you have the following dependencies installed:

- Python: 3.11+
    - gradio: 4.8.0
    - numpy: 1.26.2
    - mlflow: 2.9.1
    - pandas: 2.1.3
    - scikit-learn: 1.3.2
    - tensorflow: 2.15.0
    - tokenizers: 0.15.0
    - transforms: 4.35.2

## How to run the project

Follow the steps below to run the project on your local machine:

Execute the following commands from the project root folder:

### Clone this repository

```bash
git clone https://github.com/IgorDiasV/mlops2023
```

This link can be found on the green `Code` button above.

### Install the dependencies

```bash
pip install -r requirements.txt
```

### Environment variables

Change the `.env-example` file to `.env`.

```bash
URL_MLFLOW = "http://127.0.0.1:5000"
EXPERIMENT_NAME = ""
PATH_DATASET = ""
PATH_CLEAN_DATA = ""
PATH_TRAIN_DATA = ""
PATH_TEST_DATA = ""
PATH_WEIGHTS = ""
PATH_ENCONDER = ""
```

**Make sure you add values to null variables before continuing.**

### Run the project

Instantiate the local MLflow server
```bash
mlflow ui
```
Next, run the main pipeline. 

```bash
python .\main.py
```

### Evaluating the model

```bash
python .\interface_gradio.py
```
Navigate to `http://localhost:7860/`

## Folder Structure

The folder structure of the project is organized as follows:

```text
/
|-- unidade3/
|   |-- classification_text.py
|   |-- data_segregation.py
|   |-- fetch_data.py
|   |-- interface_gradio.py
|   |-- main.py
|   |-- pipeline.py
|   |-- preprocessing.py
|   |-- requeriments.txt
|   |-- test_predict.py
|   |-- train.py
|   |-- utils.py
|-- ...
```

## Contributors

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/IgorDiasV">
        <img src="https://github.com/IgorDiasV.png" width="100px">
        <br>
        <sub>
          <b>IgorDiasV</b>
        </sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/matheusslr">
        <img src="https://github.com/matheusslr.png" width="100px">
        <br>
        <sub>
          <b>Matheusslr</b>
        </sub>
      </a>
    </td>
  </tr>
</table>

## References

- [Bijoy Bose. (2019). BBC News Classification. Kaggle.](https://www.kaggle.com/c/learn-ai-bbc/overview)