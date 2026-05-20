import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np

# ─────────────────────────────────────────────
# PAGE CONFIG — must be the FIRST Streamlit call
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="🎬 Movie Recommender",
    page_icon="🎬",
    layout="centered"
)

# ─────────────────────────────────────────────
# CUSTOM CSS — simple, clean styling
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Background */
    .stApp { background-color: #0f0f1a; color: #f0f0f0; }

    /* Title */
    h1 { color: #f5c518 !important; font-family: Georgia, serif; }

    /* Recommendation cards */
    .movie-card {
        background: #1c1c2e;
        border-left: 4px solid #f5c518;
        border-radius: 8px;
        padding: 12px 18px;
        margin: 8px 0;
        font-size: 16px;
    }
    .movie-card span { color: #f5c518; font-weight: bold; margin-right: 8px; }

    /* Info box */
    .info-box {
        background: #1c1c2e;
        border-radius: 10px;
        padding: 16px;
        margin-top: 20px;
        font-size: 14px;
        color: #aaa;
    }

    /* Streamlit selectbox label */
    label { color: #f0f0f0 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PHASE 1: LOAD & CLEAN DATA
# ─────────────────────────────────────────────

@st.cache_data  # Cache so data loads only once (faster re-runs)
def load_data():
    """
    Load the MovieLens small dataset from the 'dataset/' folder.
    Returns cleaned movies and ratings DataFrames.
    """
    # --- Load raw CSV files ---
    movies  = pd.read_csv("dataset/movies.csv")   # movieId, title, genres
    ratings = pd.read_csv("dataset/ratings.csv")  # userId, movieId, rating, timestamp

    # --- Data Cleaning ---
    # Drop rows where any important column is missing
    movies.dropna(subset=["title"], inplace=True)
    ratings.dropna(subset=["rating"], inplace=True)

    # Remove the timestamp column — we don't need it
    ratings.drop(columns=["timestamp"], inplace=True)

    return movies, ratings


@st.cache_data
def build_model(movies, ratings):
    """
    PHASE 2: FEATURE ENGINEERING + MODEL BUILDING

    Feature Engineering:
    --------------------
    We transform raw ratings into a 2D matrix where:
        - Rows    = Movies
        - Columns = Users
        - Values  = Star rating (0 if user didn't rate that movie)

    This matrix is our "feature" for each movie. Each movie is
    represented by a vector of user ratings. Movies with similar
    rating patterns are likely to be similar in taste.

    KNN :
    ------------------------
    KNN finds the K movies whose rating vectors are most similar
    to a given movie. We use Cosine Similarity as the distance
    metric — it measures the angle between two vectors, ignoring
    magnitude, which works well for sparse rating data.
    """

    # Step 1: Keep only movies that have been rated by at least 10 users
    #         (avoids recommending obscure movies nobody has seen)
    movie_rating_counts = ratings.groupby("movieId")["rating"].count()
    popular_movie_ids   = movie_rating_counts[movie_rating_counts >= 10].index
    ratings_filtered    = ratings[ratings["movieId"].isin(popular_movie_ids)]

    # Step 2: Create pivot table → shape: (movies × users)
    #         Missing ratings are filled with 0
    movie_user_matrix = ratings_filtered.pivot_table(
        index="movieId",
        columns="userId",
        values="rating"
    ).fillna(0)

    # Step 3: Train the KNN model
    #         metric='cosine'  → uses cosine similarity
    #         algorithm='brute' → checks all neighbors (fine for small data)
    knn_model = NearestNeighbors(
        n_neighbors=6,       # 6 because 1st neighbor is the movie itself
        metric="cosine",
        algorithm="brute"
    )
    knn_model.fit(movie_user_matrix)

    # Step 4: Build a lookup table: movie title → matrix row index
    #         (so we can find a movie's row by its name)
    movie_id_to_title = movies.set_index("movieId")["title"].to_dict()
    movie_titles_in_matrix = [
        movie_id_to_title.get(mid, "Unknown") for mid in movie_user_matrix.index
    ]

    return knn_model, movie_user_matrix, movie_titles_in_matrix


def get_recommendations(movie_name, knn_model, movie_user_matrix, movie_titles):
    """
    PHASE 2 (continued): Get top 5 similar movies using KNN.

    Steps:
    1. Find the row index of the selected movie in our matrix.
    2. Ask KNN to find 6 nearest neighbors (cosine distance).
    3. Skip the first result (it's the movie itself).
    4. Return the top 5 movie names.
    """
    # Find the position of this movie in our matrix
    try:
        movie_index = movie_titles.index(movie_name)
    except ValueError:
        return []

    # Get the movie's rating vector (1 row of the matrix)
    movie_vector = movie_user_matrix.iloc[movie_index].values.reshape(1, -1)

    # Ask KNN: "which 6 movies are most similar?"
    distances, indices = knn_model.kneighbors(movie_vector, n_neighbors=6)

    # indices[0] → list of row positions of similar movies
    # distances[0] → how similar they are (lower = more similar in cosine)
    recommendations = []
    for i, idx in enumerate(indices[0]):
        if i == 0:
            continue  # Skip index 0 — that's the query movie itself
        recommendations.append(movie_titles[idx])

    return recommendations  # Returns top 5 similar movies


# ─────────────────────────────────────────────
# PHASE 3: EVALUATION METRICS (Conceptual Demo)
# ─────────────────────────────────────────────

def show_evaluation_info():
    """
    Display a conceptual explanation of evaluation metrics.

    NOTE: True evaluation of a recommender system requires knowing
    what users actually clicked/watched next (ground truth).
    Since MovieLens doesn't have that "next watch" data, we explain
    the concepts here clearly for viva/submission purposes.
    """
    with st.expander("📊 Evaluation Concepts (Click to Read)"):
        st.markdown("""
**How would we evaluate this system?**

In a real-world recommender system, we'd split the data into
train/test sets and measure:

| Metric | What it means |
|--------|---------------|
| **Precision** | Of the movies recommended, how many did the user actually like? |
| **Recall** | Of all movies the user likes, how many did we recommend? |
| **F1-Score** | Harmonic mean of Precision & Recall (balance between both) |

**Formulas:**
```
Precision = True Positives / (True Positives + False Positives)
Recall    = True Positives / (True Positives + False Negatives)
F1-Score  = 2 × (Precision × Recall) / (Precision + Recall)
```

**Confusion Matrix:**
```
               Recommended   Not Recommended
Actually Liked      TP              FN
Not Liked           FP              TN
```

**Bias vs Variance:**
- **High Bias (Underfitting):** Model is too simple, misses patterns.
  Example: recommending only blockbusters regardless of input.
- **High Variance (Overfitting):** Model memorises training data,
  fails on new users/movies.
- **Goal:** Find the sweet spot — a model that generalises well.

**For KNN specifically:**
- Small K → high variance (sensitive to noise)
- Large K → high bias (too broad recommendations)
- We use K=5 as a balanced starting point.
        """)


# ─────────────────────────────────────────────
# PHASE 4: STREAMLIT UI
# ─────────────────────────────────────────────

def main():
    # ── Header ──────────────────────────────
    st.title("🎬 Movie Recommendation System")
    st.markdown("##### Powered by KNN | MovieLens Dataset")
    st.markdown("---")

    # ── Load Data ───────────────────────────
    with st.spinner("Loading dataset..."):
        movies, ratings = load_data()

    with st.spinner("Building recommendation model..."):
        knn_model, movie_user_matrix, movie_titles = build_model(movies, ratings)

    # ── Dataset Info ────────────────────────
    col1, col2, col3 = st.columns(3)
    col1.metric("🎥 Total Movies", f"{len(movies):,}")
    col2.metric("⭐ Total Ratings", f"{len(ratings):,}")
    col3.metric("👤 Unique Users", f"{ratings['userId'].nunique():,}")

    st.markdown("---")

    # ── Movie Selection ──────────────────────
    st.subheader("🔍 Find Similar Movies")

    # Sort movie titles alphabetically for easy browsing
    sorted_titles = sorted(movie_titles)

    selected_movie = st.selectbox(
        "Select a movie you like:",
        options=sorted_titles,
        index=sorted_titles.index("Toy Story (1995)") if "Toy Story (1995)" in sorted_titles else 0
    )

    # ── Recommend Button ─────────────────────
    if st.button("🎯 Recommend Similar Movies", use_container_width=True):
        with st.spinner("Finding similar movies..."):
            recommendations = get_recommendations(
                selected_movie, knn_model, movie_user_matrix, movie_titles
            )

        if recommendations:
            st.success(f"Top 5 movies similar to **{selected_movie}**:")
            for i, movie in enumerate(recommendations, start=1):
                st.markdown(
                    f'<div class="movie-card"><span>#{i}</span>{movie}</div>',
                    unsafe_allow_html=True
                )
        else:
            st.warning("Could not find recommendations for this movie. Try another one!")

    st.markdown("---")

    # ── How It Works ────────────────────────
    with st.expander("🧠 How does this work?"):
        st.markdown("""
**Step-by-step recommendation logic:**

1. **Load Data** — Read `movies.csv` and `ratings.csv` from MovieLens.

2. **Build Matrix** — Create a *Movie × User* pivot table.
   Each cell = star rating a user gave to a movie (0 if not rated).

3. **Train KNN** — Fit a K-Nearest Neighbors model on this matrix
   using **Cosine Similarity** as the distance metric.

4. **Query** — When you pick a movie, the model finds 5 other movies
   whose rating vectors are most "angled similarly" in multi-dimensional
   user-rating space.

5. **Return Results** — Those 5 movies are your recommendations!

**Why Cosine Similarity?**
It measures the *angle* between two rating vectors — not their magnitude.
So a movie rated 5/5 by 3 people and one rated 5/5 by 300 people can
still be "similar" if their rating *patterns* match.
        """)

    # ── Evaluation Info ──────────────────────
    show_evaluation_info()

    # ── Footer ───────────────────────────────
    st.markdown("""
    <div class="info-box">
    📁 Dataset: <b>MovieLens Small</b> (grouplens.org) &nbsp;|&nbsp;
    🤖 Algorithm: <b>KNN + Cosine Similarity</b> &nbsp;|&nbsp;
    🛠️ Built with <b>Streamlit + Scikit-learn</b>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    main()
