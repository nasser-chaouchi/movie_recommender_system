import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from collections import Counter, defaultdict
import re

# --- Load data ---
@st.cache_data
def load_data():
    try:
        df_movies = pd.read_csv(
            "https://huggingface.co/datasets/nasserCha/movielens_ratings_32m/resolve/main/movies.csv",
            encoding="latin-1"
        )

        df_ratings = pd.read_csv(
            "https://huggingface.co/datasets/nasserCha/movielens_ratings_32m/resolve/main/ratings.csv",
            encoding="latin-1"
        )

        return df_movies, df_ratings

    except Exception as e:
        st.error("❌ Failed to load datasets from Hugging Face.")
        st.exception(e)
        return pd.DataFrame(), pd.DataFrame()


df_movies, df_ratings = load_data()
df_movies['genres'] = df_movies['genres'].fillna("")
df_movies['year'] = df_movies['title'].str.extract(r'\((\d{4})\)').astype(float)

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

# --- Recommendation Functions ---
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

def build_user_user_cf(df_ratings):
    df_ratings = df_ratings.copy()
    df_ratings['user_index'] = df_ratings['userId'].astype('category').cat.codes
    df_ratings['movie_index'] = df_ratings['movieId'].astype('category').cat.codes

    user_index_to_id = dict(enumerate(df_ratings['userId'].astype('category').cat.categories))
    movie_index_to_id = dict(enumerate(df_ratings['movieId'].astype('category').cat.categories))

    matrix = csr_matrix((
        df_ratings['rating'],
        (df_ratings['user_index'], df_ratings['movie_index'])
    ))

    return matrix, user_index_to_id, movie_index_to_id, df_ratings

def recommend_user_user_manual_selection(selected_titles, df_ratings, df_movies, top_n=10, like_threshold=4.0, k_neighbors=5, genre_weight=0.3):
    if not selected_titles:
        return []

    # Mapping title to movieId
    selected_ids = df_movies[df_movies['title'].isin(selected_titles)]['movieId'].tolist()

    # Build user-item matrix
    user_movie_matrix, user_index_to_id, movie_index_to_id, df_ratings = build_user_user_cf(df_ratings)
    num_movies = len(movie_index_to_id)

    # Create pseudo-user vector
    pseudo_user_vector = np.zeros(num_movies)
    for mid in selected_ids:
        if mid in movie_index_to_id.values():
            idx = list(movie_index_to_id.keys())[list(movie_index_to_id.values()).index(mid)]
            pseudo_user_vector[idx] = 1  # or 5.0 if you want to weight it higher

    # Cosine similarity
    sim_vector = cosine_similarity([pseudo_user_vector], user_movie_matrix).flatten()
    similar_users_idx = np.argsort(sim_vector)[::-1][:k_neighbors]

    # Genres
    df_movies = df_movies.copy()
    df_movies['genres'] = df_movies['genres'].fillna("")
    vectorizer = CountVectorizer(tokenizer=lambda x: x.split('|'))
    genre_matrix = vectorizer.fit_transform(df_movies['genres'])
    genre_df = pd.DataFrame(genre_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    df_movies = pd.concat([df_movies, genre_df], axis=1)

    liked_genres = df_movies[df_movies['movieId'].isin(selected_ids)][genre_df.columns].sum()
    liked_genres = liked_genres / liked_genres.sum()

    seen_movie_ids = set(selected_ids)
    score_dict = defaultdict(float)

    for neighbor_idx in similar_users_idx:
        neighbor_id = user_index_to_id[neighbor_idx]
        similarity = sim_vector[neighbor_idx]

        neighbor_likes = df_ratings[(df_ratings['userId'] == neighbor_id) & (df_ratings['rating'] >= like_threshold)]

        for _, row in neighbor_likes.iterrows():
            movie_id = row['movieId']
            if movie_id in seen_movie_ids:
                continue

            movie_row = df_movies[df_movies['movieId'] == movie_id]
            if movie_row.empty:
                continue
            movie_genres = movie_row[genre_df.columns].values.flatten()
            genre_affinity = np.dot(movie_genres, liked_genres.values)
            combined_score = (1 - genre_weight) * similarity + genre_weight * genre_affinity
            score_dict[movie_id] += combined_score

    recommended_movie_ids = sorted(score_dict, key=score_dict.get, reverse=True)[:top_n]
    titles = df_movies[df_movies['movieId'].isin(recommended_movie_ids)]['title'].tolist()
    return titles



def recommend_user_user_hybrid(user_id, df_ratings, df_movies, top_n=10, like_threshold=4.0, k_neighbors=5, genre_weight=0.3):
    df_movies = df_movies.copy()
    df_movies['genres'] = df_movies['genres'].fillna("")

    vectorizer = CountVectorizer(tokenizer=lambda x: x.split('|'))
    genre_matrix = vectorizer.fit_transform(df_movies['genres'])
    genre_df = pd.DataFrame(genre_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    df_movies = pd.concat([df_movies, genre_df], axis=1)

    user_movie_matrix, user_index_to_id, movie_index_to_id, df_ratings = build_user_user_cf(df_ratings)

    if user_id not in df_ratings['userId'].values:
        return []

    user_id_to_index = {v: k for k, v in user_index_to_id.items()}
    user_idx = user_id_to_index[user_id]

    sim_vector = cosine_similarity(user_movie_matrix[user_idx], user_movie_matrix).flatten()
    similar_users_idx = np.argsort(sim_vector)[::-1]
    similar_users_idx = [i for i in similar_users_idx if i != user_idx][:k_neighbors]

    liked_movies = df_ratings[(df_ratings['userId'] == user_id) & (df_ratings['rating'] >= like_threshold)]
    liked_movie_ids = liked_movies['movieId'].unique()
    liked_genres = df_movies[df_movies['movieId'].isin(liked_movie_ids)][genre_df.columns].sum()
    liked_genres = liked_genres / liked_genres.sum()

    seen_movie_ids = df_ratings[df_ratings['userId'] == user_id]['movieId'].unique()
    score_dict = defaultdict(float)

    for neighbor_idx in similar_users_idx:
        neighbor_id = user_index_to_id[neighbor_idx]
        similarity = sim_vector[neighbor_idx]

        neighbor_likes = df_ratings[(df_ratings['userId'] == neighbor_id) & (df_ratings['rating'] >= like_threshold)]

        for _, row in neighbor_likes.iterrows():
            movie_id = row['movieId']
            if movie_id in seen_movie_ids:
                continue

            movie_row = df_movies[df_movies['movieId'] == movie_id]
            if movie_row.empty:
                continue
            movie_genres = movie_row[genre_df.columns].values.flatten()
            genre_affinity = np.dot(movie_genres, liked_genres.values)
            combined_score = (1 - genre_weight) * similarity + genre_weight * genre_affinity
            score_dict[movie_id] += combined_score

    recommended_movie_ids = sorted(score_dict, key=score_dict.get, reverse=True)[:top_n]
    titles = df_movies[df_movies['movieId'].isin(recommended_movie_ids)]['title'].tolist()
    return titles

def recommend_popular(df_ratings, df_movies, top_n=10, min_votes=50):
    rating_stats = df_ratings.groupby('movieId')['rating'].agg(['mean', 'count']).reset_index()
    rating_stats = rating_stats[rating_stats['count'] >= min_votes]
    top_movies = rating_stats.sort_values('mean', ascending=False).head(top_n)
    return df_movies[df_movies['movieId'].isin(top_movies['movieId'])]['title'].tolist()

def recommend_for_new_user(df_movies, df_ratings, preferred_genres, top_n=10, min_votes=20):
    genre_mask = df_movies['genres'].apply(lambda g: any(gen in g for gen in preferred_genres))
    filtered_movies = df_movies[genre_mask]
    filtered_ratings = df_ratings[df_ratings['movieId'].isin(filtered_movies['movieId'])]
    return recommend_popular(filtered_ratings, filtered_movies, top_n=top_n, min_votes=min_votes)

def recommend_by_year(df_movies, df_ratings, year, top_n=10, min_votes=30):
    filtered_movies = df_movies[df_movies['year'] == float(year)]
    filtered_ratings = df_ratings[df_ratings['movieId'].isin(filtered_movies['movieId'])]
    return recommend_popular(filtered_ratings, filtered_movies, top_n=top_n, min_votes=min_votes)

def recommend_by_genre(df_movies, df_ratings, genre, top_n=10, min_votes=30):
    filtered_movies = df_movies[df_movies['genres'].str.contains(genre, na=False)]
    filtered_ratings = df_ratings[df_ratings['movieId'].isin(filtered_movies['movieId'])]
    return recommend_popular(filtered_ratings, filtered_movies, top_n=top_n, min_votes=min_votes)

def clean_title(title):
    """Remove the year from the movie title (e.g., 'The Matrix (1999)' → 'The Matrix')."""
    return re.sub(r'\s*\(\d{4}\)$', '', title)

def display_movie_list(recos, df_movies):
    """Display a list of recommended movies with title, year, and genres."""
    if not recos:
        st.info("No recommendations found.")
        return
    
    for title in recos:
        row = df_movies[df_movies['title'] == title]
        if not row.empty:
            raw_title = row.iloc[0]['title']
            title_cleaned = clean_title(raw_title)
            genres = row.iloc[0]['genres']
            year = row.iloc[0]['year']
            display_text = f"🎬 **{title_cleaned}**  \n📅 {int(year) if not pd.isna(year) else '?'}  \n🎭 _{genres}_"
        else:
            display_text = f"🎬 **{clean_title(title)}**"
        st.markdown(display_text)

# --- Streamlit UI ---
st.title("\U0001F3AC MovieLens Hybrid Recommendation System")

tab_popular, tab1, tab2, tab3, tab4, tab5  = st.tabs([
    "⭐ Popular Picks",
    "\U0001F50D By Movie", 
    "\U0001F464 Item-Item Hybrid By User", 
    "\U0001F3AF Item-Item Hybrid By Manual Selection",
    "\U0001F46B User-User Hybrid By User",
    "\U0001F3AF User-User Hybrid By Manual Selection"
])

with tab1:
    st.subheader("\U0001F50D Recommend Similar Movies")
    query = st.text_input("Start typing a movie title:")
    filtered_titles = [title for title in df_common_movies['title'].unique() if query.lower() in title.lower()]
    if filtered_titles:
        selected_movie = st.selectbox("Select a movie:", sorted(filtered_titles), key="movie_selector")
        alpha = st.slider("Balance (ratings vs. genres)", 0.0, 1.0, 0.5)
        if st.button("Recommend", key="movie_btn"):
            recos = recommend_hybrid(selected_movie, top_n=5, alpha=alpha)
            st.markdown("### \U0001F53D Recommendations:")
            display_movie_list(recos, df_common_movies)
    else:
        st.info("No matching movie found. Please type at least 3 letters.")

with tab2:
    st.subheader("\U0001F464 Personalized Recommendations")
    user_id = st.selectbox("Select a user ID:", sorted(df_ratings['userId'].unique()))
    alpha_user = st.slider("Balance (ratings vs. genres)", 0.0, 1.0, 0.5, key="user_slider")
    if st.button("Recommend", key="user_btn"):
        recos = recommend_hybrid_for_user(user_id, top_n=5, alpha=alpha_user)
        st.markdown("### \U0001F53D Recommendations:")
        display_movie_list(recos, df_common_movies)


with tab3:
    st.subheader("\U0001F3AF Recommend Based on Your Selection")
    if "selected_movies" not in st.session_state:
        st.session_state.selected_movies = []
    manual_query = st.text_input("Search for a movie to add:", key="manual_search_tab3")
    matching_manual = [title for title in df_common_movies['title'].unique() if manual_query.lower() in title.lower()]
    if matching_manual:
        manual_select = st.selectbox("Choose a movie:", sorted(matching_manual), key="manual_select_tab3")
        if st.button("\u2795 Add to selection"):
            if manual_select not in st.session_state.selected_movies:
                st.session_state.selected_movies.append(manual_select)
    if st.button("\U0001F5D1️ Clear selection"):
        st.session_state.selected_movies = []
    st.markdown("### 🎞️ Your Selection:")
    if st.session_state.selected_movies:
        for movie_title in st.session_state.selected_movies:
            row = df_movies[df_movies['title'] == movie_title]
            if not row.empty:
                title_cleaned = clean_title(movie_title)
                genres = row.iloc[0]['genres']
                year = row.iloc[0]['year']
                st.markdown(f"""
                • 🎬 **{title_cleaned}**  
                📅 {int(year) if not pd.isna(year) else "?"}  
                🎭 _{genres}_  
                """)
            else:
                st.markdown(f"• 🎬 **{clean_title(movie_title)}**")
    else:
        st.info("No movies selected yet.")
    alpha_manual = st.slider("Balance (ratings vs. genres)", 0.0, 1.0, 0.5, key="manual_slider")
    if st.button("\U0001F3AF Recommend from selection"):
        all_recos = []
        for f in st.session_state.selected_movies:
            all_recos += recommend_hybrid(f, top_n=5, alpha=alpha_manual)
        unique_recos = sorted(set(all_recos) - set(st.session_state.selected_movies))
        st.markdown("### \U0001F53D Recommendations:")
        if unique_recos:
            display_movie_list(unique_recos, df_common_movies)
        else:
            st.info("No new recommendations found (all were already selected).")

with tab4:
    st.subheader("\U0001F46B Recommendations with User-User Hybrid")
    user_id = st.selectbox("Select a user ID:", sorted(df_ratings['userId'].unique()), key="user_user_selector")
    genre_weight = st.slider("Genre weight", 0.0, 1.0, 0.3)
    if st.button("Recommend", key="user_user_btn"):
        recos = recommend_user_user_hybrid(user_id, df_ratings, df_movies, top_n=10, genre_weight=genre_weight)
        st.markdown("### \U0001F53D Recommendations:")
        display_movie_list(recos, df_common_movies)
        
        
with tab_popular:
    st.subheader("⭐ Popular Picks")

    sub_option = st.radio("Choose a recommendation method:", [
        "🎬 Top Rated Movies",
        "🎭 By Genre",
        "📅 By Year",
        "🆕 New User"
    ])

    if sub_option == "🎬 Top Rated Movies":
        if st.button("Show top movies"):
            recos = recommend_popular(df_ratings, df_movies)
            st.markdown("### 🎯 Recommendations:")
            display_movie_list(recos, df_common_movies)

    elif sub_option == "🎭 By Genre":
        all_genres = sorted(set(g for sub in df_movies['genres'].str.split('|') for g in sub if isinstance(sub, list)))
        genre = st.selectbox("Select a genre:", all_genres)
        if st.button("Recommend by Genre"):
            recos = recommend_by_genre(df_movies, df_ratings, genre)
            st.markdown("### 🎯 Recommendations:")
            display_movie_list(recos, df_common_movies)

    elif sub_option == "📅 By Year":
        year = st.selectbox("Select a year:", sorted(df_movies['year'].dropna().unique().astype(int)))
        if st.button("Recommend by Year"):
            recos = recommend_by_year(df_movies, df_ratings, year)
            st.markdown("### 🎯 Recommendations:")
            display_movie_list(recos, df_common_movies)

    elif sub_option == "🆕 New User":
        all_genres = sorted(set(g for sub in df_movies['genres'].str.split('|') for g in sub if isinstance(sub, list)))
        preferred_genres = st.multiselect("Choose your favorite genres:", all_genres)
        if st.button("Recommend for New User"):
            recos = recommend_for_new_user(df_movies, df_ratings, preferred_genres)
            st.markdown("### 🎯 Recommendations:")
            display_movie_list(recos, df_common_movies)

with tab5:
    st.subheader("\U0001F3AF Recommend Based on Your Selection")

    if "selected_movies_tab5" not in st.session_state:
        st.session_state.selected_movies_tab5 = []

    manual_query_tab5 = st.text_input("Search for a movie to add:", key="manual_search_tab5")
    matching_manual_tab5 = [
        title for title in df_common_movies['title'].unique()
        if manual_query_tab5.lower() in title.lower()
    ]

    if matching_manual_tab5:
        manual_select_tab5 = st.selectbox(
            "Choose a movie:", sorted(matching_manual_tab5), key="manual_select_tab5")
        if st.button("\u2795 Add to selection", key="add_selection_tab5"):
            if manual_select_tab5 not in st.session_state.selected_movies_tab5:
                st.session_state.selected_movies_tab5.append(manual_select_tab5)

    if st.button("\U0001F5D1️ Clear selection", key="clear_selection_tab5"):
        st.session_state.selected_movies_tab5 = []

    st.markdown("### \U0001F39E️ Your Selection:")
    if st.session_state.selected_movies_tab5:
        for movie_title in st.session_state.selected_movies_tab5:
            row = df_movies[df_movies['title'] == movie_title]
            if not row.empty:
                title_cleaned = clean_title(movie_title)
                genres = row.iloc[0]['genres']
                year = row.iloc[0]['year']
                st.markdown(f"""
                • \U0001F3AC **{title_cleaned}**  
                \U0001F4C5 {int(year) if not pd.isna(year) else "?"}  
                \U0001F3AD _{genres}_  
                """)
            else:
                st.markdown(f"• \U0001F3AC **{clean_title(movie_title)}**")
    else:
        st.info("No movies selected yet.")

    alpha_manual_tab5 = st.slider("Balance (ratings vs. genres)", 0.0, 1.0, 0.5, key="manual_slider_tab5")
    if st.button("\U0001F3AF Recommend from selection", key="recommend_tab5"):
        all_recos_tab5 = recommend_user_user_manual_selection(
            st.session_state.selected_movies_tab5,
            df_ratings,
            df_movies,
            top_n=10,
            k_neighbors=5,
            genre_weight=alpha_manual_tab5
        )
        unique_recos_tab5 = sorted(set(all_recos_tab5) - set(st.session_state.selected_movies_tab5))

        st.markdown("### \U0001F53D Recommendations:")
        if unique_recos_tab5:
            display_movie_list(unique_recos_tab5, df_common_movies)
        else:
            st.info("No new recommendations found (all were already selected).")


# with tab5:
#     st.subheader("\U0001F195 Popular Movies by Preferred Genres")
#     all_genres = sorted(set(g for sub in df_movies['genres'].str.split('|') for g in sub if isinstance(sub, list)))
#     preferred_genres = st.multiselect("Choose your favorite genres:", all_genres)
#     if st.button("Recommend", key="new_user_btn"):
#         recos = recommend_for_new_user(df_movies, df_ratings, preferred_genres)
#         st.markdown("### \U0001F53D Recommendations:")
#         display_movie_list(recos, df_common_movies)


# with tab6:
#     st.subheader("\U0001F4C5 Recommend Popular Movies by Year")
#     year = st.selectbox("Select a year:", sorted(df_movies['year'].dropna().unique().astype(int)), key="year_selector")
#     if st.button("Recommend", key="year_btn"):
#         recos = recommend_by_year(df_movies, df_ratings, year)
#         st.markdown("### \U0001F53D Recommendations:")
#         display_movie_list(recos, df_common_movies)


# with tab7:
#     st.subheader("\U0001F39E️ Recommend by Genre")
#     genre = st.selectbox("Select a genre:", all_genres, key="genre_selector")
#     if st.button("Recommend", key="genre_btn"):
#         recos = recommend_by_genre(df_movies, df_ratings, genre)
#         st.markdown("### \U0001F53D Recommendations:")
#         display_movie_list(recos, df_common_movies)


# with tab8:
#     st.subheader("\u2B50 Top Rated Popular Movies")
#     if st.button("Show top movies", key="popular_btn"):
#         recos = recommend_popular(df_ratings, df_movies)
#         st.markdown("### \U0001F53D Recommendations:")
#         display_movie_list(recos, df_common_movies)