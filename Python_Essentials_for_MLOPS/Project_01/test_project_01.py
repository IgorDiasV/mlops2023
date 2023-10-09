
from movie_recommendations import clean_title, load_files, create_widget
import pandas as pd
import ipywidgets as widgets


def test_clean_title():
    title = "test title"
    title2 = "test.// tit...le"

    assert "test title" == clean_title(title)
    assert "test title" == clean_title(title2)


def test_load_files():
    movies, ratings = load_files()

    assert isinstance(movies, pd.DataFrame)
    assert isinstance(ratings, pd.DataFrame)


def test_create_widget():
    def teste():
        return True

    widget_input, widget_output = create_widget(teste)

    isinstance(widget_input, widgets.Text)
    isinstance(widget_output, widgets.Output)
