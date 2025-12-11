import streamlit as st
import pandas as pd
import numpy as np
import ast

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


# ----------------- Data loading & preprocessing ----------------- #

@st.cache_data(show_spinner="Loading and processing movie data...")
def load_data():
    # Make sure these CSVs are in the same folder as app.py
    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")

    # Merge like in your notebook
    movies = movies.merge(credits, on="title")

    # Keep only required columns
    movies = movies[["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]]

    # Drop rows with missing values (same as notebook: movies.dropna(inplace=True))
    movies.dropna(inplace=True)

    # --- Helper functions copied from notebook ---

    def convert(obj):
        L = []
        for i in ast.literal_eval(obj):
            L.append(i["name"])
        return L

    def convert3(obj):
        L = []
        counter = 0
        for i in ast.literal_eval(obj):
            if counter != 3:
                L.append(i["name"])
                counter += 1
            else:
                break
        return L

    def fetch_director(obj):
        L = []
        for i in ast.literal_eval(obj):
            if i["job"] == "Director":
                L.append(i["name"])
                break
        return L

    # Apply conversions
    movies["genres"] = movies["genres"].apply(convert)
    movies["keywords"] = movies["keywords"].apply(convert)
    movies["cast"] = movies["cast"].apply(convert3)
    movies["crew"] = movies["crew"].apply(fetch_director)

    # Overview -> list of words
    movies["overview"] = movies["overview"].apply(lambda x: x.split())

    # Remove spaces in multi-word tokens
    movies["genres"] = movies["genres"].apply(lambda x: [i.replace(" ", "") for i in x])
    movies["keywords"] = movies["keywords"].apply(lambda x: [i.replace(" ", "") for i in x])
    movies["cast"] = movies["cast"].apply(lambda x: [i.replace(" ", "") for i in x])
    movies["crew"] = movies["crew"].apply(lambda x: [i.replace(" ", "") for i in x])

    # Create tags = overview + genres + keywords + cast + crew
    movies["tags"] = (
        movies["overview"]
        + movies["genres"]
        + movies["keywords"]
        + movies["cast"]
        + movies["crew"]
    )

    # New dataframe like in notebook
    new_df = movies[["movie_id", "title", "tags"]]

    # Join list into string and lowercase
    new_df["tags"] = new_df["tags"].apply(lambda x: " ".join(x))
    new_df["tags"] = new_df["tags"].apply(lambda x: x.lower())

    # Stemming (your nltk part)
    def stem(text):
        y = []
        for i in text.split():
            y.append(ps.stem(i))
        return " ".join(y)

    new_df["tags"] = new_df["tags"].apply(stem)

    # Vectorization and similarity
    cv = CountVectorizer(max_features=5000, stop_words="english")
    vectors = cv.fit_transform(new_df["tags"]).toarray()

    similarity = cosine_similarity(vectors)

    return new_df, similarity


movies, similarity = load_data()


# ----------------- Recommendation function ----------------- #

def recommend(movie):
    # Same logic as notebook, but return list instead of print
    try:
        movie_index = movies[movies["title"] == movie].index[0]
    except IndexError:
        return []

    distances = similarity[movie_index]
    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1],
    )[1:6]  # skip itself

    recommended_movies = []
    for i in movies_list:
        recommended_movies.append(movies.iloc[i[0]].title)

    return recommended_movies


# ----------------- Streamlit UI ----------------- #

st.title("🎬 Movie Recommender System")

selected_movie_name = st.selectbox(
    "Select a movie to get recommendations:",
    movies["title"].values,
)

if st.button("Recommend"):
    recommendations = recommend(selected_movie_name)
    if not recommendations:
        st.write("No recommendations found.")
    else:
        st.subheader("Recommended movies:")
        for name in recommendations:
            st.write("•", name)
