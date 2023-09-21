import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import ipywidgets as widgets
from IPython.display import display


def user_recs(df_all_users):
    """
        returns the ratio between the number of films and the number of unique users
    """
    qtd_movies = df_all_users["movieId"].value_counts()
    qtd_users = len(df_all_users["userId"].unique())
    return qtd_movies / qtd_users


def clean_title(title):
    """ removes unwanted characters from the title """
    title = re.sub("[^a-zA-Z0-9 ]", "", title)
    return title


def search(title):
    """ returns the 5 most similar films """
    title = clean_title(title)
    query_vec = vectorizer.transform([title])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    indices = np.argpartition(similarity, -5)[-5:]
    results = movies.iloc[indices].iloc[::-1]

    return results


def on_type_movie_input(data):
    """displays movies with similar names"""
    with movie_list:
        movie_list.clear_output()
        title = data["new"]
        if len(title) > 5:
            display(search(title))


def find_similar_movies(movie_id):
    """search for recommendation movies"""
    similar_users = ratings[(ratings["movieId"] == movie_id) &
                            (ratings["rating"] > 4)]["userId"].unique()
    similar_user_recs = ratings[(ratings["userId"].isin(similar_users)) &
                                (ratings["rating"] > 4)]["movieId"]
    similar_user_recs = similar_user_recs.value_counts() / len(similar_users)

    similar_user_recs = similar_user_recs[similar_user_recs > .10]
    all_users = ratings[(ratings["movieId"].isin(similar_user_recs.index)) &
                        (ratings["rating"] > 4)]

    all_user_recs = user_recs(all_users)
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
            results = search(title)
            movie_id = results.iloc[0]["movieId"]
            display(find_similar_movies(movie_id))


movies = pd.read_csv("ml-25m/movies.csv")
movies["clean_title"] = movies["title"].apply(clean_title)

vectorizer = TfidfVectorizer(ngram_range=(1, 2))

tfidf = vectorizer.fit_transform(movies["clean_title"])

movie_input = widgets.Text(
    value='Toy Story',
    description='Movie Title:',
    disabled=False
)
movie_list = widgets.Output()

movie_input.observe(on_type_movie_input, names='value')

display(movie_input, movie_list)

MOVIE_ID = 89745

# def find_similar_movies(movie_id):
movie = movies[movies["movieId"] == MOVIE_ID]

ratings = pd.read_csv("ml-25m/ratings.csv")

similar_users = ratings[(ratings["movieId"] == MOVIE_ID) &
                        (ratings["rating"] > 4)]["userId"].unique()

similar_user_recs = ratings[(ratings["userId"].isin(similar_users)) &
                            (ratings["rating"] > 4)]["movieId"]

similar_user_recs = similar_user_recs.value_counts() / len(similar_users)

similar_user_recs = similar_user_recs[similar_user_recs > .10]

all_users = ratings[(ratings["movieId"].isin(similar_user_recs.index)) &
                    (ratings["rating"] > 4)]

all_user_recs = user_recs(all_users)

rec_percentages = pd.concat([similar_user_recs, all_user_recs], axis=1)
rec_percentages.columns = ["similar", "all"]

rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]

rec_percentages = rec_percentages.sort_values("score", ascending=False)

rec_percentages.head(10).merge(movies, left_index=True, right_on="movieId")

movie_name_input = widgets.Text(
    value='Toy Story',
    description='Movie Title:',
    disabled=False
)
recommendation_list = widgets.Output()

movie_name_input.observe(on_type_recommendation_list, names='value')

display(movie_name_input, recommendation_list)
