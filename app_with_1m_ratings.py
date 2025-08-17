import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from collections import Counter, defaultdict
import re

# -----------------------------
# Page config & corporate CSS
# -----------------------------
st.set_page_config(page_title="MovieLens Hybrid Recommender", layout="wide")

CSS = """
<style>
html, body, [class*="css"]{
  font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Ubuntu, "Helvetica Neue", Arial, sans-serif;
}
.section-title { font-size: 1.25rem; font-weight: 700; color: #111827; margin: 8px 0 6px; }
.card {
  background: #ffffff; border:1px solid #EAECF0; border-radius:12px;
  padding:18px 20px; box-shadow:0 1px 2px rgba(16,24,40,0.06); margin-bottom:16px;
}
.card-title { font-weight:600; font-size:1.05rem; margin-bottom:8px; color:#111827; }
.caption { color:#6B7280; font-size:0.95rem; }
.hr { height:1px; background:#E5E7EB; border:0; margin:16px 0; }
.footer { color:#6B7280; font-size:0.9rem; }
.badge {
  display:inline-block; padding:4px 10px; border-radius:999px; border:1px solid #E5E7EB;
  color:#374151; background:#F9FAFB; font-weight:600; font-size:0.85rem;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_data
def load_data():
    try:
        df_movies = pd.read_csv(
            "https://huggingface.co/datasets/nasserCha/movielens_rating_1m/resolve/main/movies.dat",
            sep="::", engine="python", names=["movieId", "title", "genres"], encoding="latin-1"
        )
        df_ratings = pd.read_csv(
            "https://huggingface.co/datasets/nasserCha/movielens_rating_1m/resolve/main/ratings.dat",
            sep="::", engine="python", names=["userId", "movieId", "rating", "timestamp"], encoding="latin-1"
        )
        return df_movies, df_ratings
    except Exception as e:
        st.error("Failed to load datasets from Hugging Face.")
        st.exception(e)
        return pd.DataFrame(), pd.DataFrame()

df_movies, df_ratings = load_data()
df_movies["genres"] = df_movies["genres"].fillna("")
df_movies["year"] = df_movies["title"].str.extract(r"\((\d{4})\)").astype(float)

rated_movie_ids = df_ratings["movieId"].unique()
df_common_movies = df_movies[df_movies["movieId"].isin(rated_movie_ids)].reset_index(drop=True)

movie_id_to_index = pd.Series(df_common_movies.index, index=df_common_movies["movieId"]).to_dict()
index_to_title = pd.Series(df_common_movies["title"].values, index=df_common_movies.index).to_dict()
title_to_index = pd.Series(df_common_movies.index, index=df_common_movies["title"]).to_dict()

vectorizer = CountVectorizer(tokenizer=lambda x: x.split('|'))
genre_matrix_sparse = vectorizer.fit_transform(df_common_movies["genres"])

df_ratings_filtered = df_ratings[df_ratings["movieId"].isin(rated_movie_ids)].copy()
df_ratings_filtered["movie_index"] = df_ratings_filtered["movieId"].map(movie_id_to_index)
df_ratings_filtered["user_index"] = df_ratings_filtered["userId"].astype("category").cat.codes
movie_user_matrix = csr_matrix(
    (df_ratings_filtered["rating"], (df_ratings_filtered["movie_index"], df_ratings_filtered["user_index"]))
)

# -----------------------------
# Recommendation functions (kept)
# -----------------------------
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
    if user_id not in df_ratings["userId"].unique():
        return []
    liked = df_ratings_filtered[
        (df_ratings_filtered["userId"] == user_id) &
        (df_ratings_filtered["rating"] >= like_threshold)
    ]["movie_index"].dropna().unique()
    if len(liked) == 0:
        return []
    all_recos = []
    for idx in liked:
        recos = recommend_hybrid(index_to_title[idx], top_n=top_n, alpha=alpha)
        all_recos.extend(recos)
    reco_counts = Counter(all_recos)
    already_seen = df_ratings_filtered[df_ratings_filtered["userId"] == user_id]["movie_index"].map(index_to_title).dropna().tolist()
    filtered_recos = [title for title, _ in reco_counts.most_common() if title not in already_seen]
    return filtered_recos[:top_n]

def build_user_user_cf(df_ratings):
    df_ratings = df_ratings.copy()
    df_ratings["user_index"] = df_ratings["userId"].astype("category").cat.codes
    df_ratings["movie_index"] = df_ratings["movieId"].astype("category").cat.codes
    user_index_to_id = dict(enumerate(df_ratings["userId"].astype("category").cat.categories))
    movie_index_to_id = dict(enumerate(df_ratings["movieId"].astype("category").cat.categories))
    matrix = csr_matrix((df_ratings["rating"], (df_ratings["user_index"], df_ratings["movie_index"])))
    return matrix, user_index_to_id, movie_index_to_id, df_ratings

def recommend_user_user_manual_selection(selected_titles, df_ratings, df_movies, top_n=10, like_threshold=4.0, k_neighbors=5, genre_weight=0.3):
    if not selected_titles:
        return []
    selected_ids = df_movies[df_movies["title"].isin(selected_titles)]["movieId"].tolist()
    user_movie_matrix, user_index_to_id, movie_index_to_id, df_ratings = build_user_user_cf(df_ratings)
    num_movies = len(movie_index_to_id)

    pseudo_user_vector = np.zeros(num_movies)
    for mid in selected_ids:
        if mid in movie_index_to_id.values():
            idx = list(movie_index_to_id.keys())[list(movie_index_to_id.values()).index(mid)]
            pseudo_user_vector[idx] = 1

    sim_vector = cosine_similarity([pseudo_user_vector], user_movie_matrix).flatten()
    similar_users_idx = np.argsort(sim_vector)[::-1][:k_neighbors]

    df_movies = df_movies.copy()
    df_movies["genres"] = df_movies["genres"].fillna("")
    vectorizer = CountVectorizer(tokenizer=lambda x: x.split('|'))
    genre_matrix = vectorizer.fit_transform(df_movies["genres"])
    genre_df = pd.DataFrame(genre_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    df_movies = pd.concat([df_movies, genre_df], axis=1)

    liked_genres = df_movies[df_movies["movieId"].isin(selected_ids)][genre_df.columns].sum()
    liked_genres = liked_genres / liked_genres.sum()

    seen_movie_ids = set(selected_ids)
    score_dict = defaultdict(float)

    for neighbor_idx in similar_users_idx:
        neighbor_id = user_index_to_id[neighbor_idx]
        similarity = sim_vector[neighbor_idx]
        neighbor_likes = df_ratings[(df_ratings["userId"] == neighbor_id) & (df_ratings["rating"] >= like_threshold)]

        for _, row in neighbor_likes.iterrows():
            movie_id = row["movieId"]
            if movie_id in seen_movie_ids:
                continue
            movie_row = df_movies[df_movies["movieId"] == movie_id]
            if movie_row.empty:
                continue
            movie_genres = movie_row[genre_df.columns].values.flatten()
            genre_affinity = np.dot(movie_genres, liked_genres.values)
            combined_score = (1 - genre_weight) * similarity + genre_weight * genre_affinity
            score_dict[movie_id] += combined_score

    recommended_movie_ids = sorted(score_dict, key=score_dict.get, reverse=True)[:top_n]
    titles = df_movies[df_movies["movieId"].isin(recommended_movie_ids)]["title"].tolist()
    return titles

def recommend_user_user_hybrid(user_id, df_ratings, df_movies, top_n=10, like_threshold=4.0, k_neighbors=5, genre_weight=0.3):
    df_movies = df_movies.copy()
    df_movies["genres"] = df_movies["genres"].fillna("")
    vectorizer = CountVectorizer(tokenizer=lambda x: x.split('|'))
    genre_matrix = vectorizer.fit_transform(df_movies["genres"])
    genre_df = pd.DataFrame(genre_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    df_movies = pd.concat([df_movies, genre_df], axis=1)

    user_movie_matrix, user_index_to_id, movie_index_to_id, df_ratings = build_user_user_cf(df_ratings)
    if user_id not in df_ratings["userId"].values:
        return []

    user_id_to_index = {v: k for k, v in user_index_to_id.items()}
    user_idx = user_id_to_index[user_id]

    sim_vector = cosine_similarity(user_movie_matrix[user_idx], user_movie_matrix).flatten()
    similar_users_idx = np.argsort(sim_vector)[::-1]
    similar_users_idx = [i for i in similar_users_idx if i != user_idx][:k_neighbors]

    liked_movies = df_ratings[(df_ratings["userId"] == user_id) & (df_ratings["rating"] >= like_threshold)]
    liked_movie_ids = liked_movies["movieId"].unique()
    liked_genres = df_movies[df_movies["movieId"].isin(liked_movie_ids)][genre_df.columns].sum()
    liked_genres = liked_genres / liked_genres.sum()

    seen_movie_ids = df_ratings[df_ratings["userId"] == user_id]["movieId"].unique()
    score_dict = defaultdict(float)

    for neighbor_idx in similar_users_idx:
        neighbor_id = user_index_to_id[neighbor_idx]
        similarity = sim_vector[neighbor_idx]
        neighbor_likes = df_ratings[(df_ratings["userId"] == neighbor_id) & (df_ratings["rating"] >= like_threshold)]

        for _, row in neighbor_likes.iterrows():
            movie_id = row["movieId"]
            if movie_id in seen_movie_ids:
                continue
            movie_row = df_movies[df_movies["movieId"] == movie_id]
            if movie_row.empty:
                continue
            movie_genres = movie_row[genre_df.columns].values.flatten()
            genre_affinity = np.dot(movie_genres, liked_genres.values)
            combined_score = (1 - genre_weight) * similarity + genre_weight * genre_affinity
            score_dict[movie_id] += combined_score

    recommended_movie_ids = sorted(score_dict, key=score_dict.get, reverse=True)[:top_n]
    titles = df_movies[df_movies["movieId"].isin(recommended_movie_ids)]["title"].tolist()
    return titles

def recommend_popular(df_ratings, df_movies, top_n=10, min_votes=50):
    rating_stats = df_ratings.groupby("movieId")["rating"].agg(["mean", "count"]).reset_index()
    rating_stats = rating_stats[rating_stats["count"] >= min_votes]
    top_movies = rating_stats.sort_values("mean", ascending=False).head(top_n)
    return df_movies[df_movies["movieId"].isin(top_movies["movieId"])]["title"].tolist()

def recommend_for_new_user(df_movies, df_ratings, preferred_genres, top_n=10, min_votes=20):
    genre_mask = df_movies["genres"].apply(lambda g: any(gen in g for gen in preferred_genres))
    filtered_movies = df_movies[genre_mask]
    filtered_ratings = df_ratings[df_ratings["movieId"].isin(filtered_movies["movieId"])]
    return recommend_popular(filtered_ratings, filtered_movies, top_n=top_n, min_votes=min_votes)

def recommend_by_year(df_movies, df_ratings, year, top_n=10, min_votes=30):
    filtered_movies = df_movies[df_movies["year"] == float(year)]
    filtered_ratings = df_ratings[df_ratings["movieId"].isin(filtered_movies["movieId"])]
    return recommend_popular(filtered_ratings, filtered_movies, top_n=top_n, min_votes=min_votes)

def recommend_by_genre(df_movies, df_ratings, genre, top_n=10, min_votes=30):
    filtered_movies = df_movies[df_movies["genres"].str.contains(genre, na=False)]
    filtered_ratings = df_ratings[df_ratings["movieId"].isin(filtered_movies["movieId"])]
    return recommend_popular(filtered_ratings, filtered_movies, top_n=top_n, min_votes=min_votes)

def clean_title(title):
    return re.sub(r"\s*\(\d{4}\)$", "", title)

# Helper: render recommendations as a table
def render_table(titles: list[str], df_movies: pd.DataFrame):
    if not titles:
        st.info("No recommendations found.")
        return
    rows = []
    for t in titles:
        row = df_movies[df_movies["title"] == t]
        if not row.empty:
            year = row.iloc[0]["year"]
            rows.append({
                "Title": clean_title(row.iloc[0]["title"]),
                "Year": int(year) if pd.notna(year) else None,
                "Genres": row.iloc[0]["genres"]
            })
        else:
            rows.append({"Title": clean_title(t), "Year": None, "Genres": ""})
    df_view = pd.DataFrame(rows)
    st.dataframe(df_view, use_container_width=True, hide_index=True)


st.title("MovieLens Hybrid Recommender")
st.markdown(
    "Hybrid recommendation system combining **ratings signals** and **content (genres)**. "
    "Use the navigation on the left to explore different strategies."
)

with st.sidebar:
    st.header("Navigation")
    view = st.radio(
        "Select a view",
        [
            "Popular Picks",
            "By Movie (Item–Item Hybrid)",
            "By User (Item–Item Hybrid)",
            "Manual Selection (Item–Item Hybrid)",
            "By User (User–User Hybrid)",
            "Manual Selection (User–User Hybrid)"
        ],
        index=0
    )
    st.markdown('<hr class="hr">', unsafe_allow_html=True)
    topn = st.number_input("Top-N recommendations", min_value=3, max_value=50, value=10, step=1)
    st.markdown('<div class="caption">Data: MovieLens 1M (via Hugging Face)</div>', unsafe_allow_html=True)


if view == "Popular Picks":
    st.markdown('<div class="card"><div class="card-title">Popular Picks</div>', unsafe_allow_html=True)
    mode = st.radio("Method", ["Top Rated", "By Genre", "By Year", "For New Users"], horizontal=True)

    if mode == "Top Rated":
        if st.button("Show"):
            recos = recommend_popular(df_ratings, df_movies, top_n=topn)
            render_table(recos, df_common_movies)

    elif mode == "By Genre":
        all_genres = sorted(set(g for sub in df_movies["genres"].str.split("|") for g in (sub if isinstance(sub, list) else [])))
        genre = st.selectbox("Genre", all_genres)
        if st.button("Recommend by Genre"):
            recos = recommend_by_genre(df_movies, df_ratings, genre, top_n=topn)
            render_table(recos, df_common_movies)

    elif mode == "By Year":
        years = sorted(df_movies["year"].dropna().unique().astype(int))
        year = st.selectbox("Year", years)
        if st.button("Recommend by Year"):
            recos = recommend_by_year(df_movies, df_ratings, year, top_n=topn)
            render_table(recos, df_common_movies)

    elif mode == "For New Users":
        all_genres = sorted(set(g for sub in df_movies["genres"].str.split("|") for g in (sub if isinstance(sub, list) else [])))
        prefs = st.multiselect("Preferred genres", all_genres)
        if st.button("Recommend for New Users"):
            recos = recommend_for_new_user(df_movies, df_ratings, prefs, top_n=topn)
            render_table(recos, df_common_movies)

    st.markdown('</div>', unsafe_allow_html=True)

elif view == "By Movie (Item–Item Hybrid)":
    st.markdown('<div class="card"><div class="card-title">Similar to a Selected Movie</div>', unsafe_allow_html=True)
    query = st.text_input("Movie title")
    filtered = [t for t in df_common_movies["title"].unique() if query.lower() in t.lower()] if query else []
    if filtered:
        selected = st.selectbox("Select", sorted(filtered))
        alpha = st.slider("Balance (ratings vs. genres)", 0.0, 1.0, 0.5)
        if st.button("Recommend", key="btn_movie"):
            recos = recommend_hybrid(selected, top_n=topn, alpha=alpha)
            render_table(recos, df_common_movies)
    else:
        st.caption("Type part of a movie title to search.")
    st.markdown('</div>', unsafe_allow_html=True)

elif view == "By User (Item–Item Hybrid)":
    st.markdown('<div class="card"><div class="card-title">Personalized (User → Item–Item Hybrid)</div>', unsafe_allow_html=True)
    user = st.selectbox("User ID", sorted(df_ratings["userId"].unique()))
    alpha = st.slider("Balance (ratings vs. genres)", 0.0, 1.0, 0.5, key="alpha_item_user")
    if st.button("Recommend", key="btn_item_user"):
        recos = recommend_hybrid_for_user(user, top_n=topn, alpha=alpha)
        render_table(recos, df_common_movies)
    st.markdown('</div>', unsafe_allow_html=True)

elif view == "Manual Selection (Item–Item Hybrid)":
    st.markdown('<div class="card"><div class="card-title">Manual Selection → Item–Item Hybrid</div>', unsafe_allow_html=True)

    
    if "sel_movies_item" not in st.session_state:
        st.session_state.sel_movies_item = []

    search = st.text_input("Search a movie to add", key="search_item_sel")
    pool = [t for t in df_common_movies["title"].unique() if search.lower() in t.lower()] if search else []
    if pool:
        choice = st.selectbox("Choose a movie", sorted(pool), key="choice_item_sel")
        if st.button("Add to selection", key="add_item_sel"):
            if choice not in st.session_state.sel_movies_item:
                st.session_state.sel_movies_item.append(choice)
                st.rerun()

    c1, c2 = st.columns([3, 1])
    with c1:
        st.caption("Current selection")
        if st.session_state.sel_movies_item:
            html_badges = " ".join(f'<span class="badge">{m}</span>' for m in st.session_state.sel_movies_item)
            st.markdown(html_badges, unsafe_allow_html=True)
        else:
            st.caption("—")
    with c2:
        if st.button("Clear", key="clear_item_sel"):
            st.session_state.sel_movies_item = []
            st.rerun()

    alpha = st.slider("Balance (ratings vs. genres)", 0.0, 1.0, 0.5, key="alpha_item_manual")
    if st.button("Recommend from selection", key="btn_item_sel_reco"):
        combined = []
        for t in st.session_state.sel_movies_item:
            combined += recommend_hybrid(t, top_n=topn, alpha=alpha)
        unique = sorted(set(combined) - set(st.session_state.sel_movies_item))
        render_table(unique, df_common_movies)

    st.markdown('</div>', unsafe_allow_html=True)


elif view == "By User (User–User Hybrid)":
    st.markdown('<div class="card"><div class="card-title">User–User Hybrid</div>', unsafe_allow_html=True)
    user = st.selectbox("User ID", sorted(df_ratings["userId"].unique()), key="user_user_id")
    w = st.slider("Genre weight", 0.0, 1.0, 0.3)
    if st.button("Recommend", key="btn_user_user"):
        recos = recommend_user_user_hybrid(user, df_ratings, df_movies, top_n=topn, genre_weight=w)
        render_table(recos, df_common_movies)
    st.markdown('</div>', unsafe_allow_html=True)

elif view == "Manual Selection (User–User Hybrid)":
    st.markdown('<div class="card"><div class="card-title">Manual Selection → User–User Hybrid</div>', unsafe_allow_html=True)
    if "sel_movies_user" not in st.session_state:
        st.session_state.sel_movies_user = []
    search = st.text_input("Search a movie to add", key="search_user_sel")
    pool = [t for t in df_common_movies["title"].unique() if search.lower() in t.lower()] if search else []
    if pool:
        choice = st.selectbox("Choose a movie", sorted(pool), key="choice_user_sel")
        if st.button("Add to selection", key="add_user_sel"):
            if choice not in st.session_state.sel_movies_user:
                st.session_state.sel_movies_user.append(choice)

    c1, c2 = st.columns([3, 1])
    with c1:
        st.caption("Current selection")
        if st.session_state.sel_movies_user:
            html_badges = " ".join(
                [f'<span class="badge">{m}</span>' for m in st.session_state.sel_movies_user]
            )
            st.markdown(html_badges, unsafe_allow_html=True)
        else:
            st.caption("—")
    with c2:
        if st.button("Clear", key="clear_user_sel"):
            st.session_state.sel_movies_user = []
            st.rerun()

    w = st.slider("Genre weight", 0.0, 1.0, 0.3, key="w_user_sel")
    if st.button("Recommend from selection", key="btn_user_sel_reco"):
        recos = recommend_user_user_manual_selection(
            st.session_state.sel_movies_user, df_ratings, df_movies,
            top_n=topn, k_neighbors=5, genre_weight=w
        )
        unique = sorted(set(recos) - set(st.session_state.sel_movies_user))
        render_table(unique, df_common_movies)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<hr class="hr">', unsafe_allow_html=True)
st.markdown(
    '<div class="footer">'
    'Author: <strong>Nasser Chaouchi</strong> &nbsp;|&nbsp; '
    '<a href="https://www.linkedin.com/in/nasser-chaouchi/" target="_blank">LinkedIn</a> &nbsp;|&nbsp; '
    '<a href="https://github.com/nasser-chaouchi" target="_blank">GitHub</a>'
    '</div>',
    unsafe_allow_html=True
)