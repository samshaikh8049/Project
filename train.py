"""
Training and Evaluation Pipeline for Movie Recommendation System.
Performs:
1. Data loading and preprocessing
2. Train-test split on ratings for offline collaborative filtering evaluation
3. Fits TruncatedSVD Matrix Factorization and measures test RMSE and MAE
4. Fits TF-IDF Content-Based engine
5. Evaluates Top-N recommendation sample outputs
6. Serializes model artifacts to models/ directory
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.decomposition import TruncatedSVD

from src.data_loader import get_preprocessed_dataset
from src.recommender import MovieRecommenderEngine, MODELS_DIR

def evaluate_svd_collaborative(ratings, n_components=30):
    """
    Offline evaluation of SVD collaborative filtering using 80/20 train/test split.
    Calculates Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE).
    """
    print("\n--- Evaluating Collaborative Filtering Model (SVD) ---")
    train_ratings, test_ratings = train_test_split(ratings, test_size=0.2, random_state=42)
    
    # Filter to users and movies present in train set
    user_item_matrix = train_ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    
    # Latent factor matrix factorization
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    user_factors = svd.fit_transform(user_item_matrix.values)
    movie_factors = svd.components_
    
    # Reconstructed predictions matrix
    predicted_matrix = np.dot(user_factors, movie_factors)
    pred_df = pd.DataFrame(predicted_matrix, index=user_item_matrix.index, columns=user_item_matrix.columns)
    
    # Evaluate on test set
    y_true = []
    y_pred = []
    
    for _, row in test_ratings.iterrows():
        u = row['userId']
        m = row['movieId']
        if u in pred_df.index and m in pred_df.columns:
            predicted_rating = pred_df.loc[u, m]
            # Clip predictions to valid rating scale [0.5, 5.0]
            predicted_rating = max(0.5, min(5.0, predicted_rating))
            y_true.append(row['rating'])
            y_pred.append(predicted_rating)
            
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    print(f"Test Set Evaluated Ratings: {len(y_true)}")
    print(f"Collaborative SVD Test RMSE: {rmse:.4f}")
    print(f"Collaborative SVD Test MAE:  {mae:.4f}")
    
    return {"rmse": round(float(rmse), 4), "mae": round(float(mae), 4), "samples": len(y_true)}

def main():
    print("=" * 60)
    print(" MOVIE RECOMMENDATION SYSTEM - TRAINING & EVALUATION ")
    print("=" * 60)
    start_time = time.time()
    
    # 1. Load & Preprocess Data
    print("\n[Step 1/5] Loading & Preprocessing Dataset...")
    movies_df, ratings, tags, links = get_preprocessed_dataset()
    print(f"Processed {len(movies_df)} movies and {len(ratings)} user ratings.")
    
    # 2. Evaluate SVD Collaborative Filtering
    print("\n[Step 2/5] Performing Offline Model Evaluation...")
    eval_metrics = evaluate_svd_collaborative(ratings, n_components=30)
    
    # 3. Fit Production Recommendation Engine
    print("\n[Step 3/5] Fitting Production Multi-Model Recommendation Engine...")
    engine = MovieRecommenderEngine(movies_df, ratings)
    engine.fit_all()
    
    # 4. Test Sample Inferences
    print("\n[Step 4/5] Testing Sample Inferences...")
    test_queries = ["Toy Story", "The Dark Knight", "Inception"]
    for query in test_queries:
        print(f"\n--- Recommendations for '{query}' (Hybrid Engine) ---")
        recs, target = engine.recommend_hybrid(query, top_n=5)
        if target is not None:
            print(f"Target: {target['title']} | Genres: {target['genres']}")
            for idx, r in recs.iterrows():
                print(f"  -> {r['title']} ({r['year']}) | Genres: {r['genres']} | Rating: {r['vote_average']} | Match: {r['match_score']}")
        else:
            print(f"Query '{query}' not found.")
            
    # 5. Serialize Artifacts
    print("\n[Step 5/5] Saving Serialized Artifacts for Fast Web Inference...")
    engine.save_artifacts()
    
    # Save training summary metrics
    summary_path = os.path.join(MODELS_DIR, "training_metrics.joblib")
    import joblib
    joblib.dump(eval_metrics, summary_path)
    
    elapsed = round(time.time() - start_time, 2)
    print("\n" + "=" * 60)
    print(f" Training and Evaluation Pipeline Completed in {elapsed}s! ")
    print("=" * 60)

if __name__ == "__main__":
    main()

