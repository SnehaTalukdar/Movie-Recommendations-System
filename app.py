# Movie Recommendation System
# © 2025 Sneha Talukdar
# Developed during ElevateLabs Internship
# For educational and non-commercial use only

import streamlit as st
import pandas as pd
import numpy as np
import warnings

# -------------------- SETTINGS --------------------
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Movie Recommendation System"
)

# -------------------- TITLE --------------------
st.title("🎬 Movie Recommendation System")

st.markdown(
    "Get top 5 similar movies using collaborative filtering"
)

# -------------------- LOAD DATA --------------------
@st.cache_data(show_spinner=False)
def load_data():

    # Smaller dataset for speed + stability
    ratings = pd.read_csv(
        "data/ratings_small.csv"
    )

    movies = pd.read_csv(
        "data/movies_metadata.csv",
        usecols=["id", "title"],
        low_memory=False
    )

    # Clean movie IDs
    movies["id"] = pd.to_numeric(
        movies["id"],
        errors="coerce"
    )

    movies.dropna(
        subset=["id", "title"],
        inplace=True
    )

    # Merge datasets
    merged = ratings.merge(
        movies,
        left_on="movieId",
        right_on="id"
    )

    merged = merged[
        ["userId", "title", "rating"]
    ].dropna()

    return merged


df = load_data()

# -------------------- SAFETY CHECK --------------------
if df.empty:
    st.error("Dataset not loaded properly.")
    st.stop()

# -------------------- FILTER POPULAR MOVIES --------------------
movie_counts = df["title"].value_counts()

popular_movies = movie_counts[
    movie_counts >= 100
].index

df = df[
    df["title"].isin(popular_movies)
]

# -------------------- MOVIE LIST --------------------
movie_list = sorted(
    df["title"].unique()
)

selected_movie = st.selectbox(
    "Choose a movie:",
    movie_list
)

# -------------------- USER-MOVIE MATRIX --------------------
@st.cache_data(show_spinner=False)
def create_matrix(data):

    matrix = data.pivot_table(
        index="userId",
        columns="title",
        values="rating"
    )

    return matrix


user_movie_matrix = create_matrix(df)

# -------------------- RECOMMENDATION FUNCTION --------------------
def get_similar_movies(movie_title):

    if movie_title not in user_movie_matrix.columns:
        return pd.DataFrame()

    target_ratings = user_movie_matrix[
        movie_title
    ]

    # Pearson Correlation Similarity
    similar_movies = user_movie_matrix.corrwith(
        target_ratings
    )

    corr_df = pd.DataFrame(
        similar_movies,
        columns=["Similarity"]
    )

    # Remove invalid values
    corr_df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    corr_df.dropna(inplace=True)

    # Count ratings
    rating_counts = df.groupby(
        "title"
    )["rating"].count()

    corr_df["rating_count"] = rating_counts

    # -------------------- FILTERS --------------------

    # Keep movies with enough ratings
    filtered = corr_df[
        corr_df["rating_count"] >= 100
    ]

    # Remove self similarity
    filtered = filtered[
        filtered["Similarity"] < 0.999
    ]

    # Remove negative similarity
    filtered = filtered[
        filtered["Similarity"] > 0
    ]

    # Remove suspicious near-perfect matches
    filtered = filtered[
        filtered["Similarity"] < 0.95
    ]

    # Sort recommendations
    recommendations = filtered.sort_values(
        "Similarity",
        ascending=False
    ).head(5)

    return recommendations

# -------------------- BUTTON --------------------
if st.button("Recommend Similar Movies"):

    recommendations = get_similar_movies(
        selected_movie
    )

    st.subheader("⭐ Top Similar Movies")

    if not recommendations.empty:

        recommendations = recommendations.reset_index()

        for _, row in recommendations.iterrows():

            st.write(
                f"🎬 {row['title']} → Similarity Score: {round(row['Similarity'], 3)}"
            )

    else:

        st.warning(
            "No similar movies found. Try another movie."
        )

# -------------------- FOOTER --------------------
st.markdown("---")

st.markdown(
    """
    <div style='text-align: center; font-size: 14px;'>
    Developed by <b>Sneha Talukdar</b> during ElevateLabs Internship, 2025<br>
    For educational and non-commercial use only.
    </div>
    """,
    unsafe_allow_html=True
)