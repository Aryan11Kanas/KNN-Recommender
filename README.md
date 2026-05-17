# 🎬 Movie Recommendation System
### Beginner-Friendly | Python + Streamlit + KNN | MovieLens Dataset

---

## 📌 Project Overview

This is a simple, beginner-friendly **Movie Recommendation System** built using:
- **Python** — core programming language
- **Pandas** — data loading and manipulation
- **Scikit-learn** — KNN model and cosine similarity
- **Streamlit** — interactive web UI

The system recommends **5 movies similar to the one you select**, based on how other users rated them.

---

## 📁 Project Structure

```
movie-recommender/
│
├── app.py               ← Main Streamlit application (all logic here)
├── requirements.txt     ← Python dependencies
├── README.md            ← This file
│
└── dataset/
    ├── movies.csv       ← Movie titles and genres (MovieLens)
    └── ratings.csv      ← User ratings (MovieLens)
```

---

## 📥 Step 1: Download the Dataset

1. Go to: https://grouplens.org/datasets/movielens/latest/
2. Download **ml-latest-small.zip** (about 1 MB)
3. Unzip it
4. Copy **`movies.csv`** and **`ratings.csv`** into the **`dataset/`** folder

Your folder should look like:
```
dataset/
├── movies.csv
└── ratings.csv
```

---

## ⚙️ Step 2: Install Dependencies

Make sure you have **Python 3.8+** installed.

Open your terminal/command prompt in the project folder and run:

```bash
pip install -r requirements.txt
```

This installs: `streamlit`, `pandas`, `scikit-learn`, `numpy`

---

## ▶️ Step 3: Run the App

```bash
streamlit run app.py
```

Your browser will automatically open at: **http://localhost:8501**

---

## 🧪 How It Works (Step by Step)

### Phase 1 — Data Cleaning
```
movies.csv  →  drop missing titles
ratings.csv →  drop missing ratings, remove timestamp column
```

### Phase 2 — Feature Engineering
We create a **Movie × User matrix** (pivot table):

```
           User1  User2  User3  User4 ...
Toy Story    5      0      4      3   ...
Jumanji      0      3      0      5   ...
GoldenEye    4      4      0      0   ...
```
- Rows = Movies
- Columns = Users
- Values = Star rating (0 = not rated)

This matrix is the **feature representation** of each movie.

### Phase 3 — KNN Model
```python
KNN(n_neighbors=6, metric='cosine', algorithm='brute')
```
- We train KNN on the movie-user matrix
- To recommend movies for "Toy Story", KNN finds 5 other movies
  whose rating vectors have the smallest cosine distance

### Phase 4 — Streamlit UI
- Dropdown to select a movie
- Button to get recommendations
- Cards showing top 5 results

---

## 📐 Key Concepts Explained

### 🔷 Feature Engineering
Converting raw data (ratings) into a format useful for ML.
Here: transforming userId-movieId-rating rows → a 2D matrix.

### 🔷 KNN (K-Nearest Neighbors)
Finds the K most similar items to a given item.
No training in the traditional sense — it memorises the data
and computes distances at query time.

### 🔷 Cosine Similarity
Measures the angle between two vectors:
```
cosine_similarity = (A · B) / (|A| × |B|)
```
- Result = 1 → vectors point same direction (very similar)
- Result = 0 → vectors are perpendicular (not similar)

Works well for sparse data (lots of zeros).

### 🔷 Precision
```
Precision = TP / (TP + FP)
```
"Of everything we recommended, how much was actually good?"

### 🔷 Recall
```
Recall = TP / (TP + FN)
```
"Of everything that was actually good, how much did we recommend?"

### 🔷 F1-Score
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
Balances precision and recall into a single number.

### 🔷 Confusion Matrix
```
               Recommended    Not Recommended
Actually Liked    TP (✓)          FN (✗)
Not Liked         FP (✗)          TN (✓)
```

### 🔷 Bias vs Variance
| | Bias | Variance |
|---|---|---|
| **High** | Underfitting (too simple) | Overfitting (too complex) |
| **Symptom** | Misses real patterns | Fails on new data |
| **KNN fix** | Decrease K | Increase K |

---

## ❗ Common Beginner Mistakes & Fixes

| Mistake | Fix |
|---------|-----|
| `FileNotFoundError: movies.csv` | Make sure files are inside the `dataset/` folder |
| App shows no movies in dropdown | Dataset files may be corrupt — re-download |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` first |
| Streamlit not found | Run `pip install streamlit` |
| App doesn't open in browser | Manually go to `http://localhost:8501` |
| Recommendations look wrong | Normal — sparse data means approximate results |

---

## 🎓 Viva Q&A Cheat Sheet

**Q: Why KNN and not a neural network?**
A: KNN is simple, interpretable, and works well for small datasets. No training time needed.

**Q: Why cosine similarity and not Euclidean distance?**
A: Cosine similarity ignores the magnitude of ratings (a power user rating 100 movies
doesn't dominate over one who rated 10). It focuses on the *pattern* of ratings.

**Q: What is the movie-user matrix?**
A: A 2D table where rows are movies, columns are users, and values are star ratings.
It represents each movie as a vector in "user space."

**Q: Why filter movies with fewer than 10 ratings?**
A: To avoid recommending obscure movies. Sparse rating vectors give unreliable similarity scores.

**Q: What does `@st.cache_data` do?**
A: It tells Streamlit to cache the result of a function so it doesn't re-run every time
the user interacts with the UI — makes the app much faster.

---

## 📊 Dataset Info

| File | Columns | Description |
|------|---------|-------------|
| `movies.csv` | movieId, title, genres | ~9,742 movies |
| `ratings.csv` | userId, movieId, rating, timestamp | ~100,836 ratings from ~610 users |

Source: [GroupLens Research](https://grouplens.org/datasets/movielens/latest/)

---

## 🚀 Quick Start (Copy-Paste)

```bash
# 1. Clone or download the project
cd movie-recommender

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add dataset files to dataset/ folder
#    (download from grouplens.org/datasets/movielens/latest/)

# 4. Run the app
streamlit run app.py
```

---

*Built for college submission — simple, clean, and educational.*
