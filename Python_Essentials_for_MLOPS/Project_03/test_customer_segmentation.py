import pandas as pd
from credit_card_customer_segmentation import load_file


def test_load_file():
    customers = load_file()

    assert isinstance(customers, pd.DataFrame)