import pandas as pd
import numpy as np
from credit_card_customer_segmentation import load_file, preprocess_customer_data
from credit_card_customer_segmentation import find_optimal_clusters, train_kmeans_model


def test_load_file():
    customers = load_file()

    assert isinstance(customers, pd.DataFrame)


def test_preprocess_customer_data():
    df = load_file()

    preprocessed_data, preprocess_data_scaled = preprocess_customer_data(df)

    first_value = -0.1654055800960332

    assert preprocess_data_scaled[0][0] == first_value
    assert isinstance(preprocessed_data, pd.DataFrame)


def test_optimal_clusters():
    df = load_file()
    preprocessed_data, _ = preprocess_customer_data(df)
    optimal_clusters = find_optimal_clusters(preprocessed_data)

    assert isinstance(optimal_clusters, list)


def test_train_kmeans_model():
    df = load_file()
    _, preprocess_data_scaled = preprocess_customer_data(df)
    train_model = train_kmeans_model(preprocess_data_scaled)

    assert isinstance(train_model, np.ndarray)

