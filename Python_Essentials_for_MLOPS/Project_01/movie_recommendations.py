"""get five movie recommendations based on a chosen movie"""
import re
import logging
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import ipywidgets as widgets
from IPython.display import display

DATA_DIR = "Python_Essentials_for_MLOPS/Project_01/ml-25m/"

logging.basicConfig(level=logging.INFO)

def load_files():
    """loads the dataframes used in the code"""
    df_movies = pd.read_csv(DATA_DIR + "movies.csv")
    df_ratings = pd.read_csv(DATA_DIR + "ratings.csv")
    return df_movies, df_ratings

def calculate_user_recs(df_all_users: pd.DataFrame) -> float:
    """Calculates the ratio of movies to unique users."""
    qtd_movies = df_all_users["movieId"].value_counts()
    qtd_users = len(df_all_users["userId"].unique())

    try:
        result = qtd_movies / qtd_users
    except ZeroDivisionError:
        result = 0

    return result


def clean_title(title: str) -> str:
    """ Remove from the title any character
        that is not a letter or a number. """
    title = re.sub("[^a-zA-Z0-9 ]", "", title)
    return title


def search_similar_movies(title: str) -> pd.DataFrame:
    """ returns the 5 most similar films """
    title = clean_title(title)
    query_vec = vectorizer.transform([title])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    indices = np.argpartition(similarity, -5)[-5:]
    results = movies.iloc[indices].iloc[::-1]
    return results

def create_widget(function_input_change):
    """Creates and returns the widget."""
    widget_input = widgets.Text(
        value='Toy Story',
        description='Movie Title:',
        disabled=False
    )
    widget_output = widgets.Output()
    widget_input.observe(function_input_change, names='value')
    return widget_input, widget_output

def on_type_movie_input(data):
    """displays movies with similar names"""

    with movie_list:
        movie_list.clear_output()
        title = data["new"]
        if len(title) > 5:
            logging.info("starting the search for similar films")
            display(search_similar_movies(title))
            logging.info("Finished the search for similar films")


def find_similar_movies(movie_id: int) -> pd.DataFrame:
    """search for recommendation movies"""

    similar_users = ratings[(ratings["movieId"] == movie_id) &
                            (ratings["rating"] > 4)]["userId"].unique()
    similar_user_recs = ratings[(ratings["userId"].isin(similar_users)) &
                                (ratings["rating"] > 4)]["movieId"]

    try:
        similar_user_recs = similar_user_recs.value_counts() / len(similar_users)
    except ZeroDivisionError:
        logging.error("No similar users found")

    similar_user_recs = similar_user_recs[similar_user_recs > .10]
    all_users = ratings[(ratings["movieId"].isin(similar_user_recs.index)) &
                        (ratings["rating"] > 4)]
    all_user_recs = calculate_user_recs(all_users)

    rec_percentages = pd.concat([similar_user_recs, all_user_recs], axis=1)
    rec_percentages.columns = ["similar", "all"]
    rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]
    rec_percentages = rec_percentages.sort_values("score", ascending=False)
 
    top_recpercentages = rec_percentages.head(10)
    similar_movies = top_recpercentages.merge(movies,
                                              left_index=True,
                                              right_on="movieId")
    similar_movies = similar_movies[["score", "title", "genres"]]
    return similar_movies


def on_type_recommendation_list(data):
    """displays movie recommendation"""

    with recommendation_list:
        recommendation_list.clear_output()
        title = data["new"]
        if len(title) > 5:
            logging.info("starting the search for similar films")
            results = search_similar_movies(title)
            movie_id = results.iloc[0]["movieId"]
            display(find_similar_movies(movie_id))
            logging.info("Finished the search for recommended films")


if __name__ == "__main__":

    movies, ratings = load_files()

    movies["clean_title"] = movies["title"].apply(clean_title)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2))

    tfidf = vectorizer.fit_transform(movies["clean_title"])

    movie_input, movie_list = create_widget(on_type_movie_input)

    display(movie_input, movie_list)

    MOVIE_ID = 89745

    movie = movies[movies["movieId"] == MOVIE_ID]

    similar_users = ratings[(ratings["movieId"] == MOVIE_ID) &
                            (ratings["rating"] > 4)]["userId"].unique()

    similar_user_recs = ratings[(ratings["userId"].isin(similar_users)) &
                                (ratings["rating"] > 4)]["movieId"]

    try:
        similar_user_recs = similar_user_recs.value_counts() / len(similar_users)
    except ZeroDivisionError:
        logging.error("no similar users found")

    similar_user_recs = similar_user_recs[similar_user_recs > .10]

    all_users = ratings[(ratings["movieId"].isin(similar_user_recs.index)) &
                        (ratings["rating"] > 4)]

    all_user_recs = calculate_user_recs(all_users)

    rec_percentages = pd.concat([similar_user_recs, all_user_recs], axis=1)
    rec_percentages.columns = ["similar", "all"]

    rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]

    rec_percentages = rec_percentages.sort_values("score", ascending=False)

    rec_percentages.head(10).merge(movies, left_index=True, right_on="movieId")

    movie_name_input, recommendation_list = create_widget(on_type_recommendation_list)
    
    display(movie_name_input, recommendation_list)
