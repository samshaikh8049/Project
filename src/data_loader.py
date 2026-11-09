"""
Data loading and preprocessing module for Movie Recommendation System.
Loads MovieLens dataset and prepares content and collaborative feature representations.
"""

import os
import re
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "ml-latest-small")

def load_raw_data(data_dir=DATA_DIR):
    """Load raw MovieLens CSV files."""
    movies_path = os.path.join(data_dir, "movies.csv")
    ratings_path = os.path.join(data_dir, "ratings.csv")
    tags_path = os.path.join(data_dir, "tags.csv")
    links_path = os.path.join(data_dir, "links.csv")

    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)
    tags = pd.read_csv(tags_path) if os.path.exists(tags_path) else pd.DataFrame(columns=["movieId", "tag"])
    links = pd.read_csv(links_path) if os.path.exists(links_path) else pd.DataFrame(columns=["movieId", "tmdbId"])

    return movies, ratings, tags, links

def extract_year(title):
    """Extract release year from movie title string."""
    match = re.search(r'\((\d{4})\)', str(title))
    return int(match.group(1)) if match else np.nan

def clean_title(title):
    """Remove year and strip excess whitespace from movie title."""
    return re.sub(r'\s*\(\d{4}\)\s*', '', str(title)).strip()

def preprocess_movies(movies, ratings, tags):
    """
    Clean movie metadata, aggregate tags, and compute weighted Bayesian ratings.
    """
    df = movies.copy()
    
    # Extract year and clean title
    df['year'] = df['title'].apply(extract_year)
    df['clean_title'] = df['title'].apply(clean_title)
    
    # Format genres: replace '|' with ' '
    df['genres_list'] = df['genres'].apply(lambda x: [] if x == "(no genres listed)" else x.split('|'))
    df['genres_str'] = df['genres'].apply(lambda x: "" if x == "(no genres listed)" else x.replace('|', ' '))
    
    # Aggregate user tags per movie
    if not tags.empty:
        tags_agg = tags.groupby('movieId')['tag'].apply(lambda t: " ".join(t.dropna().astype(str).str.lower())).reset_index()
        df = df.merge(tags_agg, on='movieId', how='left')
        df['tag'] = df['tag'].fillna('')
    else:
        df['tag'] = ''
        
    # Create metadata soup for NLP TF-IDF (repeating genres 2x for appropriate genre weight)
    df['soup'] = df['genres_str'] + " " + df['genres_str'] + " " + df['tag']
    df['soup'] = df['soup'].str.strip()
    
    # Rating statistics
    rating_stats = ratings.groupby('movieId').agg(
        vote_count=('rating', 'count'),
        vote_average=('rating', 'mean')
    ).reset_index()
    
    df = df.merge(rating_stats, on='movieId', how='left')
    df['vote_count'] = df['vote_count'].fillna(0).astype(int)
    df['vote_average'] = df['vote_average'].fillna(0.0).round(2)
    
    # Calculate IMDb Weighted Rating (Bayesian average)
    # WR = (v / (v + m)) * R + (m / (v + m)) * C
    C = ratings['rating'].mean()
    m = df['vote_count'].quantile(0.70)  # 70th percentile cutoff
    
    def weighted_rating(x, m=m, C=C):
        v = x['vote_count']
        R = x['vote_average']
        if v + m == 0:
            return 0.0
        return round((v / (v + m) * R) + (m / (v + m) * C), 2)
    
    df['weighted_rating'] = df.apply(weighted_rating, axis=1)
    
    return df

def get_preprocessed_dataset():
    """High-level helper to load and return preprocessed dataframes."""
    movies, ratings, tags, links = load_raw_data()
    movies_df = preprocess_movies(movies, ratings, tags)
    return movies_df, ratings, tags, links

if __name__ == "__main__":
    movies_df, ratings, tags, links = get_preprocessed_dataset()
    print("Preprocessing completed successfully!")
    print(f"Total Movies: {len(movies_df)}")
    print(f"Top 5 Weighted Rated Movies:\n{movies_df.nlargest(5, 'weighted_rating')[['title', 'vote_count', 'vote_average', 'weighted_rating']]}")

