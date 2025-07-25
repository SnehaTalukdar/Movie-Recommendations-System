import streamlit as st
import pandas as pd

# Setting the Streamlit page config
st.set_page_config(page_title="Movie Recommendation System")

# Loading and merging datasets
@st.cache_data
def load_data():
    try:
        # Loading ratings and movie metadata
        ratings = pd.read_csv("data/ratings_small.csv")
        movies = pd.read_csv("data/movies_metadata.csv", low_memory=False)

        # Converting 'id' in movies to numeric (to match movieId)
        movies['id'] = pd.to_numeric(movies['id'], errors='coerce')
        merged = pd.merge(ratings, movies, left_on='movieId', right_on='id')

        # Dropping rows with missing titles
        merged = merged[['userId', 'title', 'rating']].dropna()

        return merged

    except FileNotFoundError:
        st.error(" Required dataset not found! Make sure 'ratings_small.csv' and 'movies_metadata.csv' are in the data folder.")
        return pd.DataFrame()

df = load_data()

# If data is loaded
if not df.empty:
    st.title(" Movie Recommendation System")
    st.markdown("Get top 5 movie recommendations based on your favorite movie (collaborative filtering).")

    movie_list = df["title"].dropna().unique()
    selected_movie = st.selectbox(" Choose a movie you like:", sorted(movie_list))

    # Build user-movie matrix
    user_movie_matrix = df.pivot_table(index='userId', columns='title', values='rating')

    # Recommendation logic
    def get_similar_movies(movie_title, min_ratings=10):
        if movie_title not in user_movie_matrix.columns:
            return pd.DataFrame()

        target_ratings = user_movie_matrix[movie_title]
        similar_movies = user_movie_matrix.corrwith(target_ratings)

        corr_df = pd.DataFrame(similar_movies, columns=["Correlation"])
        corr_df.dropna(inplace=True)

        rating_counts = df.groupby("title")["rating"].count()
        corr_df["rating_count"] = rating_counts

        filtered = corr_df[corr_df["rating_count"] >= min_ratings]
        return filtered.sort_values("Correlation", ascending=False).head(5)

    if st.button(" Recommend Similar Movies"):
        recommendations = get_similar_movies(selected_movie)
        if not recommendations.empty:
            st.subheader(" Top 5 Similar Movies:")
            st.dataframe(recommendations.reset_index())
        else:
            st.warning("No similar movies found or not enough ratings.")

else:
    st.warning(" Dataset not loaded. Please check the file paths and try again.")


st.markdown("---")
st.markdown(
    "<div style='text-align: center; font-size: 14px;'>"
    "Developed by <b>Sneha Talukdar</b> during the online <b>ElevateLabs Internship, 2025</b><br>"
    " For educational and non-commercial use only."
    "</div>",
    unsafe_allow_html=True
)
