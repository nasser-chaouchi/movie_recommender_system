import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from collections import Counter

# --- Load data ---
@st.cache_data
def load_data():
    try:
        df_movies = pd.read_csv("https://huggingface.co/datasets/nasserCha/movielens_ratings_32m/resolve/main/movies.csv")
        df_ratings = pd.read_csv("https://huggingface.co/datasets/nasserCha/movielens_ratings_32m/resolve/main/ratings.csv")
        return df_movies, df_ratings
    except Exception as e:
        st.error("❌ Failed to load datasets from Hugging Face.")
        st.exception(e)
        return pd.DataFrame(), pd.DataFrame()  # empty fallback

df_movies, df_ratings = load_data()

# --- Preprocessing ---
df_movies['genres'] = df_movies['genres'].fillna("")
rated_movie_ids = df_ratings['movieId'].unique()
df_common_movies = df_movies[df_movies['movieId'].isin(rated_movie_ids)].reset_index(drop=True)

movie_id_to_index = pd.Series(df_common_movies.index, index=df_common_movies['movieId']).to_dict()
index_to_title = pd.Series(df_common_movies['title'].values, index=df_common_movies.index).to_dict()
title_to_index = pd.Series(df_common_movies.index, index=df_common_movies['title']).to_dict()

vectorizer = CountVectorizer(tokenizer=lambda x: x.split('|'))
genre_matrix_sparse = vectorizer.fit_transform(df_common_movies['genres'])

df_ratings_filtered = df_ratings[df_ratings['movieId'].isin(rated_movie_ids)].copy()
df_ratings_filtered['movie_index'] = df_ratings_filtered['movieId'].map(movie_id_to_index)
df_ratings_filtered['user_index'] = df_ratings_filtered['userId'].astype("category").cat.codes
movie_user_matrix = csr_matrix((
    df_ratings_filtered['rating'],
    (df_ratings_filtered['movie_index'], df_ratings_filtered['user_index'])
))

# --- Hybrid Recommendation ---
def recommend_hybrid(movie_title, top_n=5, alpha=0.5):
    if movie_title not in title_to_index:
        return []
    idx = title_to_index[movie_title]
    genre_sim = cosine_similarity(genre_matrix_sparse[idx], genre_matrix_sparse).flatten()
    rating_sim = cosine_similarity(movie_user_matrix[idx], movie_user_matrix).flatten()
    combined_score = alpha * rating_sim + (1 - alpha) * genre_sim
    top_indices = combined_score.argsort()[::-1]
    top_indices = [i for i in top_indices if i != idx][:top_n]
    return [index_to_title[i] for i in top_indices]

# --- Hybrid Recommendation for User ---
def recommend_hybrid_for_user(user_id, top_n=10, alpha=0.5, like_threshold=4.0):
    if user_id not in df_ratings['userId'].unique():
        return []
    liked = df_ratings_filtered[
        (df_ratings_filtered['userId'] == user_id) &
        (df_ratings_filtered['rating'] >= like_threshold)
    ]['movie_index'].dropna().unique()
    if len(liked) == 0:
        return []
    all_recos = []
    for idx in liked:
        recos = recommend_hybrid(index_to_title[idx], top_n=top_n, alpha=alpha)
        all_recos.extend(recos)
    reco_counts = Counter(all_recos)
    already_seen = df_ratings_filtered[df_ratings_filtered['userId'] == user_id]['movie_index'].map(index_to_title).dropna().tolist()
    filtered_recos = [title for title, _ in reco_counts.most_common() if title not in already_seen]
    return filtered_recos[:top_n]

# --- Streamlit UI ---
st.title("🎬 MovieLens Hybrid Recommendation System")

tab1, tab2, tab3 = st.tabs(["🔍 By Movie", "👤 By User", "🎯 Manual Selection"])

# --- Tab 1: By Movie ---
with tab1:
    st.subheader("🔍 Recommend Similar Movies")

    query = st.text_input("Start typing a movie title:")
    filtered_titles = [title for title in df_common_movies['title'].unique() if query.lower() in title.lower()]
    
    if filtered_titles:
        selected_movie = st.selectbox("Select a movie:", sorted(filtered_titles), key="movie_selector")
        alpha = st.slider("Balance (ratings vs. genres)", 0.0, 1.0, 0.5)
        if st.button("Recommend", key="movie_btn"):
            recos = recommend_hybrid(selected_movie, top_n=5, alpha=alpha)
            st.markdown("### 🔽 Recommendations:")
            st.write(recos)
    else:
        st.info("No matching movie found. Please type at least 3 letters.")

# --- Tab 2: By User ---
with tab2:
    st.subheader("👤 Personalized Recommendations")
    user_id = st.selectbox("Select a user ID:", sorted(df_ratings['userId'].unique()))
    alpha_user = st.slider("Balance (ratings vs. genres)", 0.0, 1.0, 0.5, key="user_slider")
    if st.button("Recommend", key="user_btn"):
        recos = recommend_hybrid_for_user(user_id, top_n=5, alpha=alpha_user)
        st.markdown("### 🔽 Recommendations:")
        st.write(recos)

# --- Tab 3: Manual Film Selection ---
with tab3:
    st.subheader("🎯 Recommend Based on Your Selection")

    if "selected_movies" not in st.session_state:
        st.session_state.selected_movies = []

    manual_query = st.text_input("Search for a movie to add:", key="manual_search")
    matching_manual = [title for title in df_common_movies['title'].unique() if manual_query.lower() in title.lower()]
    if matching_manual:
        manual_select = st.selectbox("Choose a movie:", sorted(matching_manual), key="manual_select")
        if st.button("➕ Add to selection"):
            if manual_select not in st.session_state.selected_movies:
                st.session_state.selected_movies.append(manual_select)

    if st.button("🗑️ Clear selection"):
        st.session_state.selected_movies = []

    st.markdown("### 🎞️ Your Selection:")
    st.write(st.session_state.selected_movies)

    alpha_manual = st.slider("Balance (ratings vs. genres)", 0.0, 1.0, 0.5, key="manual_slider")
    if st.button("🎯 Recommend from selection"):
        all_recos = []
        for f in st.session_state.selected_movies:
            all_recos += recommend_hybrid(f, top_n=5, alpha=alpha_manual)
        st.markdown("### 🔽 Recommendations:")
        st.write(sorted(set(all_recos)))
