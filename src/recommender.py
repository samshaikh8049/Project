"""
Recommendation Engine Module.
Implements:
1. Content-Based Filtering (TF-IDF + Cosine Similarity on genres and tags)
2. Item-Based Collaborative Filtering (Cosine similarity on User-Item Rating Matrix)
3. Latent Factor Matrix Factorization (Truncated SVD)
4. Hybrid Recommendation Engine (Content + Collaborative + Bayesian Rating Blend)
5. Preference-Based Filter (Genre, Year, Rating criteria)
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import difflib

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

class MovieRecommenderEngine:
    def __init__(self, movies_df, ratings_df):
        self.movies_df = movies_df.reset_index(drop=True)
        self.ratings_df = ratings_df
        
        # Mapping helpers
        self.title_to_idx = {title.lower(): idx for idx, title in enumerate(self.movies_df['title'])}
        self.clean_title_to_idx = {title.lower(): idx for idx, title in enumerate(self.movies_df['clean_title'])}
        self.movieId_to_idx = {movieId: idx for idx, movieId in enumerate(self.movies_df['movieId'])}
        self.idx_to_movieId = {idx: movieId for idx, movieId in enumerate(self.movies_df['movieId'])}
        
        # Matrices initialized lazily or via fit()
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.content_sim = None
        
        self.ratings_matrix = None
        self.item_sim = None
        self.svd_model = None
        self.svd_movie_factors = None
        self.svd_sim = None
        
    def fit_content_based(self):
        """Fit TF-IDF on metadata soup and compute content similarity matrix."""
        print("Fitting Content-Based TF-IDF Model...")
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.movies_df['soup'].fillna(''))
        self.content_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        print("Content similarity matrix computed:", self.content_sim.shape)
        
    def fit_collaborative(self, n_components=30):
        """
        Build User-Item matrix and fit Item-Item Collaborative + TruncatedSVD models.
        """
        print("Building User-Item Matrix for Collaborative Filtering...")
        # Filter movies with at least 5 ratings for robust collaborative similarity
        ratings_summary = self.ratings_df.groupby('movieId').size()
        popular_movie_ids = set(ratings_summary[ratings_summary >= 3].index)
        
        filtered_ratings = self.ratings_df[self.ratings_df['movieId'].isin(popular_movie_ids)]
        
        # Pivot table: rows = movieId, columns = userId
        pivot = filtered_ratings.pivot(index='movieId', columns='userId', values='rating').fillna(0)
        self.ratings_matrix = pivot
        
        # Item-Item Cosine Similarity
        print("Computing Item-Item Cosine Similarity...")
        item_sim_raw = cosine_similarity(pivot.values)
        self.item_sim = pd.DataFrame(item_sim_raw, index=pivot.index, columns=pivot.index)
        
        # Latent Factor Matrix Factorization with TruncatedSVD
        print(f"Fitting TruncatedSVD ({n_components} components)...")
        n_comp = min(n_components, pivot.shape[1] - 1, pivot.shape[0] - 1)
        self.svd_model = TruncatedSVD(n_components=n_comp, random_state=42)
        self.svd_movie_factors = self.svd_model.fit_transform(pivot.values)
        self.svd_sim = pd.DataFrame(
            cosine_similarity(self.svd_movie_factors),
            index=pivot.index,
            columns=pivot.index
        )
        print("Collaborative models fitted successfully.")

    def fit_all(self):
        """Fit all recommendation models."""
        self.fit_content_based()
        self.fit_collaborative()

    def find_movie_idx(self, query):
        """Fuzzy match a user query to find closest movie index."""
        query_clean = query.strip().lower()
        
        # 1. Exact match in full title
        if query_clean in self.title_to_idx:
            return self.title_to_idx[query_clean]
            
        # 2. Exact match in clean title
        if query_clean in self.clean_title_to_idx:
            return self.clean_title_to_idx[query_clean]
            
        # Check inverted article pattern: 'the dark knight' -> 'dark knight, the'
        for article in ['the ', 'a ', 'an ']:
            if query_clean.startswith(article):
                inverted = query_clean[len(article):].strip() + f", {article.strip()}"
                if inverted in self.clean_title_to_idx:
                    return self.clean_title_to_idx[inverted]
                if inverted in self.title_to_idx:
                    return self.title_to_idx[inverted]
                # Also try matching prefix
                matches = self.movies_df[self.movies_df['title'].str.lower().str.startswith(inverted)]
                if not matches.empty:
                    return matches.index[0]

        # 3. Fuzzy match against clean titles
        clean_titles = list(self.movies_df['clean_title'].str.lower())
        closest = difflib.get_close_matches(query_clean, clean_titles, n=1, cutoff=0.6)
        if closest:
            return self.clean_title_to_idx[closest[0]]

        # 4. Fuzzy match against all titles
        all_titles = list(self.movies_df['title'].str.lower())
        closest = difflib.get_close_matches(query_clean, all_titles, n=1, cutoff=0.6)
        if closest:
            return self.title_to_idx[closest[0]]
            
        # 5. Partial substring search
        matches = self.movies_df[self.movies_df['title'].str.lower().str.contains(query_clean, regex=False)]
        if not matches.empty:
            return matches.index[0]
            
        return None

    def recommend_content(self, movie_title, top_n=5):
        """
        Content-Based Recommendation using TF-IDF on genres and user tags.
        """
        idx = self.find_movie_idx(movie_title)
        if idx is None:
            return pd.DataFrame(), None
            
        target_movie = self.movies_df.iloc[idx]
        sim_scores = list(enumerate(self.content_sim[idx]))
        
        # Sort by similarity, exclude the queried movie itself
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = [s for s in sim_scores if s[0] != idx][:top_n]
        
        rec_indices = [s[0] for s in sim_scores]
        scores = [round(float(s[1]), 4) for s in sim_scores]
        
        results = self.movies_df.iloc[rec_indices].copy()
        results['match_score'] = [f"{int(s * 100)}%" for s in scores]
        results['similarity_raw'] = scores
        results['model_type'] = "Content-Based (TF-IDF & Genres)"
        
        return results[['movieId', 'title', 'year', 'genres', 'vote_average', 'vote_count', 'weighted_rating', 'match_score', 'similarity_raw', 'model_type']], target_movie

    def recommend_collaborative(self, movie_title, top_n=5, method='svd'):
        """
        Collaborative Filtering Recommendation based on user rating patterns.
        Method can be 'svd' (latent factor) or 'cosine' (item-item).
        """
        idx = self.find_movie_idx(movie_title)
        if idx is None:
            return pd.DataFrame(), None
            
        target_movie = self.movies_df.iloc[idx]
        target_movieId = target_movie['movieId']
        
        matrix = self.svd_sim if method == 'svd' else self.item_sim
        if matrix is None or target_movieId not in matrix.index:
            # Fallback to content-based if movie has insufficient ratings
            print(f"Movie {movie_title} not in collaborative matrix. Falling back to content.")
            recs, _ = self.recommend_content(movie_title, top_n=top_n)
            if not recs.empty:
                recs['model_type'] = "Content-Based (Collab Cold-Start Fallback)"
            return recs, target_movie
            
        sim_series = matrix.loc[target_movieId].drop(target_movieId, errors='ignore')
        top_similar_ids = sim_series.sort_values(ascending=False).head(top_n)
        
        results = self.movies_df[self.movies_df['movieId'].isin(top_similar_ids.index)].copy()
        # Maintain sorted order
        results['similarity_raw'] = results['movieId'].map(top_similar_ids).round(4)
        results = results.sort_values(by='similarity_raw', ascending=False)
        results['match_score'] = results['similarity_raw'].apply(lambda s: f"{int(max(0, s) * 100)}%")
        results['model_type'] = f"Collaborative Filtering ({'Latent SVD' if method == 'svd' else 'Item-Item'})"
        
        return results[['movieId', 'title', 'year', 'genres', 'vote_average', 'vote_count', 'weighted_rating', 'match_score', 'similarity_raw', 'model_type']], target_movie

    def recommend_hybrid(self, movie_title, top_n=5, alpha=0.55):
        """
        Hybrid Recommendation: Blends Content-Based similarity, SVD Collaborative similarity,
        and Bayesian weighted rating.
        Score = alpha * Sim_content + (1 - alpha) * Sim_collab + 0.1 * Normalized_WR
        """
        idx = self.find_movie_idx(movie_title)
        if idx is None:
            return pd.DataFrame(), None
            
        target_movie = self.movies_df.iloc[idx]
        target_movieId = target_movie['movieId']
        
        # 1. Content scores
        content_scores = self.content_sim[idx].copy()
        
        # 2. Collaborative scores
        if self.svd_sim is not None and target_movieId in self.svd_sim.index:
            collab_series = self.svd_sim.loc[target_movieId]
            # Map into full movie index array
            collab_scores = np.zeros(len(self.movies_df))
            for m_id, sim in collab_series.items():
                if m_id in self.movieId_to_idx:
                    collab_scores[self.movieId_to_idx[m_id]] = max(0.0, sim)
        else:
            collab_scores = content_scores.copy()
            alpha = 1.0  # Cold-start movie fallback
            
        # 3. Normalized weighted ratings (0 to 1 scale)
        max_wr = self.movies_df['weighted_rating'].max()
        min_wr = self.movies_df['weighted_rating'].min()
        norm_wr = (self.movies_df['weighted_rating'] - min_wr) / (max_wr - min_wr + 1e-6)
        
        # Blended hybrid score
        hybrid_scores = (alpha * content_scores) + ((1.0 - alpha) * collab_scores) + (0.08 * norm_wr.values)
        
        # Exclude queried movie
        hybrid_scores[idx] = -1.0
        
        top_indices = np.argsort(hybrid_scores)[::-1][:top_n]
        results = self.movies_df.iloc[top_indices].copy()
        raw_scores = hybrid_scores[top_indices]
        
        # Normalize display percentage
        norm_display = (raw_scores / np.max(raw_scores) * 98.0).astype(int) if len(raw_scores) > 0 else []
        results['match_score'] = [f"{min(99, max(50, s))}%" for s in norm_display]
        results['similarity_raw'] = np.round(raw_scores, 4)
        results['model_type'] = "Hybrid Engine (Content + Collaborative + Bayesian)"
        
        return results[['movieId', 'title', 'year', 'genres', 'vote_average', 'vote_count', 'weighted_rating', 'match_score', 'similarity_raw', 'model_type']], target_movie

    def recommend_by_preferences(self, selected_genres=None, min_rating=3.5, min_votes=10, year_range=(1970, 2024), top_n=10):
        """
        Filter and rank movies based on user preferences.
        """
        df = self.movies_df.copy()
        
        # Filter by year
        df = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]
        
        # Filter by votes and rating
        df = df[(df['vote_count'] >= min_votes) & (df['vote_average'] >= min_rating)]
        
        # Filter by genres
        if selected_genres:
            genre_mask = df['genres_list'].apply(lambda g_list: any(g in g_list for g in selected_genres))
            df = df[genre_mask]
            
        # Rank by Bayesian weighted rating
        results = df.sort_values(by='weighted_rating', ascending=False).head(top_n).copy()
        results['match_score'] = (results['weighted_rating'] / 5.0 * 100).round().astype(int).astype(str) + "%"
        results['model_type'] = "Preference-Based Discovery"
        
        return results[['movieId', 'title', 'year', 'genres', 'vote_average', 'vote_count', 'weighted_rating', 'match_score', 'model_type']]

    def save_artifacts(self, models_dir=MODELS_DIR):
        """Save computed matrices and preprocessed metadata to disk for instant loading."""
        os.makedirs(models_dir, exist_ok=True)
        print(f"Saving artifacts to {models_dir}...")
        joblib.dump(self.movies_df, os.path.join(models_dir, "movies_df.joblib"))
        joblib.dump(self.content_sim, os.path.join(models_dir, "content_sim.joblib"))
        joblib.dump(self.svd_sim, os.path.join(models_dir, "svd_sim.joblib"))
        joblib.dump(self.item_sim, os.path.join(models_dir, "item_sim.joblib"))
        print("Artifacts saved successfully!")

    @classmethod
    def load_from_artifacts(cls, models_dir=MODELS_DIR, ratings_df=None):
        """Instantiate recommender engine directly from precomputed serialized artifacts."""
        movies_df_path = os.path.join(models_dir, "movies_df.joblib")
        content_sim_path = os.path.join(models_dir, "content_sim.joblib")
        svd_sim_path = os.path.join(models_dir, "svd_sim.joblib")
        item_sim_path = os.path.join(models_dir, "item_sim.joblib")
        
        if not (os.path.exists(movies_df_path) and os.path.exists(content_sim_path)):
            return None
            
        movies_df = joblib.load(movies_df_path)
        dummy_ratings = ratings_df if ratings_df is not None else pd.DataFrame(columns=['userId', 'movieId', 'rating'])
        engine = cls(movies_df, dummy_ratings)
        engine.content_sim = joblib.load(content_sim_path)
        
        if os.path.exists(svd_sim_path):
            engine.svd_sim = joblib.load(svd_sim_path)
        if os.path.exists(item_sim_path):
            engine.item_sim = joblib.load(item_sim_path)
            
        return engine
