
from movie_recommendations import clean_title, load_files
import pandas as pd


def test_clean_title():
    title = "test title"
    title2 = "test.// tit...le"

    assert "test title" == clean_title(title)
    assert "test title" == clean_title(title2)


def test_load_files():
    movies, ratings = load_files()

    assert isinstance(movies, pd.DataFrame)
    assert isinstance(ratings, pd.DataFrame)
