
from movie_recommendations import clean_title
import pandas as pd


def test_clean_title():
    title = "test title"
    title2 = "test.// tit...le"

    assert "test title" == clean_title(title)
    assert "test title" == clean_title(title2)


def test_load_files():
    movies = pd.read_csv("Python_Essentials_for_MLOPS/Project_01/ml-25m/movies.csv")
    ratings = pd.read_csv("Python_Essentials_for_MLOPS/Project_01/ml-25m/ratings.csv")

    assert isinstance(movies, pd.DataFrame)
    assert isinstance(ratings, pd.DataFrame)
