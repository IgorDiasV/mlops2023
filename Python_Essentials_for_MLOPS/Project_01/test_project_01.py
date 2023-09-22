
from movie_recommendations import clean_title


def test_clean_title():
    title = "test title"
    title2 = "test.// tit...le"

    assert "test title" == clean_title(title)
    assert "test title" == clean_title(title2)