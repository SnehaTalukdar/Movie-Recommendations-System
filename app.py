# Movie Recommendation System
# © 2025 Sneha Talukdar
# Developed during the ElevateLabs Internship
# For educational and non-commercial use only

import streamlit as st
import pandas as pd

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Movie Recommendation System")

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    try:
        ratings = pd.read_csv("data/ratings_small.csv")
        movies = pd.read_csv("data/movies_metadata.csv", low_memory=False)

        movies['id'] = pd.to_numeric(movies['id'], errors='coerce')
        merged = pd.merge(ratings, movies, left_on='movieId', right_on='id')

        merged = merged[['userId', 'title', 'rating']].dropna()
        return merged

    except FileNotFoundError:
        st.error("Dataset not found! Check file paths.")
        return pd.DataFrame()

df = load_data()

# -------------------- MAIN APP --------------------
if not df.empty:

    st.title("Movie Recommendation System")
    st.markdown("Get top 5 similar movies using collaborative filtering")

    movie_list = df["title"].dropna().unique()
    selected_movie = st.selectbox("Choose a movie:", sorted(movie_list))

    # -------------------- USER-MOVIE MATRIX --------------------
    user_movie_matrix = df.pivot_table(index='userId', columns='title', values='rating')

    # -------------------- RECOMMENDATION FUNCTION --------------------
    def get_similar_movies(movie_title):

        if movie_title not in user_movie_matrix.columns:
            return pd.DataFrame()

        target_ratings = user_movie_matrix[movie_title]

        # Correlation-based similarity
        similar_movies = user_movie_matrix.corrwith(target_ratings)

        corr_df = pd.DataFrame(similar_movies, columns=["Correlation"])
        corr_df = corr_df.dropna()

        # Count ratings per movie
        rating_counts = df.groupby("title")["rating"].count()
        corr_df["rating_count"] = rating_counts

        # -------------------- FIXES --------------------
        # 1. Ensure enough data support
        filtered = corr_df[corr_df["rating_count"] >= 50]

        # 2. Remove fake perfect similarity
        filtered = filtered[filtered["Correlation"] < 0.9999]

        # 3. Remove negative/noisy correlations
        filtered = filtered[filtered["Correlation"] > 0]

        # Sort results
        return filtered.sort_values("Correlation", ascending=False).head(5)

    # -------------------- UI ACTION --------------------
    if st.button("Recommend Similar Movies"):

        recommendations = get_similar_movies(selected_movie)

        if not recommendations.empty:

            st.subheader("Top 5 Similar Movies with Scores:")

            recommendations = recommendations.reset_index()

            for _, row in recommendations.iterrows():
                st.write(
                    f"{row['title']} → Similarity Score: {round(row['Correlation'], 3)}"
                )

        else:
            st.warning("No similar movies found. Try another movie.")

else:
    st.warning("Dataset not loaded. Please check file paths.")

# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; font-size: 14px;'>"
    "Developed by <b>Sneha Talukdar</b> during ElevateLabs Internship, 2025<br>"
    "For educational and non-commercial use only."
    "</div>",
    unsafe_allow_html=True
)