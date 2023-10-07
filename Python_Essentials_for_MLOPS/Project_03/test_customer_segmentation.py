import pandas as pd

def test_load_file():
    customers = pd.read_csv("Python_Essentials_for_MLOPS/Project_03/customer_segmentation.csv")

    assert isinstance(customers, pd.DataFrame)