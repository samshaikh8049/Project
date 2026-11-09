"""
CineMatch AI - Intelligent Movie Recommendation System
ElevateLabs AI/ML Internship Project
Built with Python, Scikit-Learn, Pandas, and Streamlit
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="CineMatch AI - Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #FF4B4B, #FF8533);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .sub-title {
        color: #888888;
        font-size: 1.05rem;
        margin-top: 0px;
        margin-bottom: 1.5rem;
    }
    .movie-card {
        background-color: #1e2129;
        border-radius: 12px;
        padding: 16px;
        border: 1px solid #2d3139;
        box-shadow: 0 4px 6px rgba(0,0,0,0.15);
        margin-bottom: 12px;
        transition: transform 0.2s;
    }
    .badge {
        display: inline-block;
        padding: 3px 9px;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 12px;
        margin-right: 4px;
        margin-bottom: 4px;
        background-color: #313745;
        color: #e0e6ed;
    }
    .match-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.9rem;
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
    }
    .stat-box {
        background-color: #181b22;
        border-radius: 10px;
        padding: 12px;
        text-align: center;
        border: 1px solid #262b35;
    }
</style>
""", unsafe_allow_html=True)

# Imports from src
from src.data_loader import get_preprocessed_dataset
from src.recommender import MovieRecommenderEngine, MODELS_DIR

@st.cache_resource(show_spinner="Loading models & datasets...")
def load_system():
    """Load or initialize models and preprocessed data."""
    movies_df, ratings, tags, links = get_preprocessed_dataset()
    engine = MovieRecommenderEngine.load_from_artifacts(MODELS_DIR, ratings_df=ratings)
    if engine is None:
        engine = MovieRecommenderEngine(movies_df, ratings)
        engine.fit_all()
        engine.save_artifacts()
    return engine, movies_df, ratings

engine, movies_df, ratings = load_system()

# Sidebar
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=500&auto=format&fit=crop&q=60", use_container_width=True)
    st.markdown("### 🎬 CineMatch AI")
    st.markdown("**Domain:** Machine Learning / NLP")
    st.markdown("**Dataset:** MovieLens 100K Verified")
    st.markdown("**Author:** ElevateLabs AI/ML Intern")
    st.markdown("---")
    
    st.markdown("#### ⚙️ Recommendation Engine")
    model_choice = st.radio(
        "Default Engine Algorithm:",
        ("Hybrid (Recommended)", "Content-Based (TF-IDF)", "Collaborative SVD (Latent)", "Item-Item Collaborative"),
        index=0
    )
    
    top_n = st.slider("Number of Recommendations:", min_value=3, max_value=20, value=6, step=1)
    
    if "Hybrid" in model_choice:
        alpha_slider = st.slider(
            "Hybrid Weight (Content vs Collaborative):",
            min_value=0.0,
            max_value=1.0,
            value=0.55,
            step=0.05,
            help="Higher values favor genre/story similarity. Lower values favor user rating patterns."
        )
    else:
        alpha_slider = 0.55
        
    st.markdown("---")
    report_path = "Movie_Recommendation_System_Project_Report.pdf"
    if os.path.exists(report_path):
        with open(report_path, "rb") as f:
            st.download_button(
                label="📄 Download 2-Page Project Report",
                data=f,
                file_name="ElevateLabs_Movie_Recommendation_Report.pdf",
                mime="application/pdf",
                use_container_width=True
            )

# Header Section
st.markdown("<h1 class='main-title'>🎬 CineMatch AI: Next-Gen Movie Recommender</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>ElevateLabs Internship Project | Hybrid Machine Learning & Content Filtering Engine</p>", unsafe_allow_html=True)

# Metrics Strip
col1, col2, col3, col4 = st.columns(4)
col1.metric("Curated Movies", f"{len(movies_df):,}")
col2.metric("User Ratings", f"{len(ratings):,}")
col3.metric("Active Reviewers", f"{ratings['userId'].nunique():,}")
col4.metric("Avg Rating", f"{ratings['rating'].mean():.2f} / 5.0")

st.markdown("---")

# Main Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Movie-to-Movie Recommender",
    "🎨 Personalized Discovery",
    "📊 Exploratory Data Analysis (EDA)",
    "🧠 Algorithms & Architecture"
])

# ----------------- TAB 1: MOVIE RECOMMANDATIONS -----------------
with tab1:
    st.subheader("Discover Movies Similar to Your Favorites")
    st.markdown("Type a movie title or choose from one of the popular suggestions below:")
    
    sample_movies = ["Inception", "The Dark Knight", "Toy Story", "Pulp Fiction", "Interstellar", "Fight Club", "Matrix, The"]
    
    selected_preset = st.selectbox("Quick Picks:", ["-- Select a popular movie --"] + sample_movies)
    
    # Auto-fill or custom input
    all_titles_list = sorted(list(movies_df['title'].values))
    default_text = selected_preset if selected_preset != "-- Select a popular movie --" else "Inception"
    
    query_title = st.text_input("Search Movie Title:", value=default_text, placeholder="e.g. Inception, Avatar, Shawshank...")
    
    btn_recommend = st.button("🚀 Generate Recommendations", type="primary", use_container_width=True)
    
    if btn_recommend or query_title:
        with st.spinner("Analyzing movie vector similarities..."):
            if "Hybrid" in model_choice:
                recs, target = engine.recommend_hybrid(query_title, top_n=top_n, alpha=alpha_slider)
            elif "Content-Based" in model_choice:
                recs, target = engine.recommend_content(query_title, top_n=top_n)
            elif "Latent" in model_choice:
                recs, target = engine.recommend_collaborative(query_title, top_n=top_n, method='svd')
            else:
                recs, target = engine.recommend_collaborative(query_title, top_n=top_n, method='cosine')
                
        if target is None or recs.empty:
            st.error(f"Movie '{query_title}' not found in the database. Please try another title!")
        else:
            # Display Target Movie Card
            st.markdown(f"""
            <div style="background-color: #1a2332; border-left: 5px solid #3b82f6; padding: 14px 18px; border-radius: 8px; margin-bottom: 20px;">
                <h4 style="margin: 0; color: #60a5fa;">🎯 Input Target: {target['title']}</h4>
                <p style="margin: 4px 0 0 0; color: #cbd5e1; font-size: 0.95rem;">
                    <b>Year:</b> {int(target['year']) if pd.notnull(target['year']) else 'N/A'} | 
                    <b>Genres:</b> {target['genres']} | 
                    <b>Avg Rating:</b> ⭐ {target['vote_average']} / 5 ({target['vote_count']} votes) |
                    <b>IMDb Weighted:</b> 🏆 {target['weighted_rating']}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.write(f"### Top {len(recs)} Recommendations ({recs.iloc[0]['model_type']}):")
            
            # Display grid of recommendation cards
            num_cols = 3 if top_n >= 3 else top_n
            cols = st.columns(num_cols)
            
            for i, (_, row) in enumerate(recs.iterrows()):
                col = cols[i % num_cols]
                with col:
                    genre_badges = "".join([f"<span class='badge'>{g}</span>" for g in row['genres'].split('|')])
                    year_str = int(row['year']) if pd.notnull(row['year']) else "N/A"
                    
                    st.markdown(f"""
                    <div class="movie-card">
                        <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 8px;">
                            <span class="match-pill">{row['match_score']} Match</span>
                            <span style="color: #f59e0b; font-weight: bold; font-size: 0.95rem;">⭐ {row['vote_average']}</span>
                        </div>
                        <h4 style="margin: 4px 0 6px 0; font-size: 1.05rem; line-height: 1.3; min-height: 42px;">{row['title']}</h4>
                        <p style="margin: 0 0 8px 0; color: #94a3b8; font-size: 0.85rem;">Year: <b>{year_str}</b> | Votes: {int(row['vote_count'])}</p>
                        <div style="margin-top: 6px; min-height: 50px;">
                            {genre_badges}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

# ----------------- TAB 2: PERSONALIZED DISCOVERY -----------------
with tab2:
    st.subheader("🎨 Personalized Discovery by Taste & Criteria")
    st.markdown("Customize your preferred genres, minimum rating, and era to uncover hidden cinematic masterpieces.")
    
    col_a, col_b = st.columns([1, 2])
    
    with col_a:
        all_genres = ["Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary", 
                      "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", 
                      "Sci-Fi", "Thriller", "War", "Western"]
        
        pref_genres = st.multiselect("Select Preferred Genres:", all_genres, default=["Sci-Fi", "Action", "Thriller"])
        
        pref_rating = st.slider("Minimum Average Rating (1-5):", min_value=2.0, max_value=4.8, value=3.8, step=0.1)
        
        pref_votes = st.slider("Minimum Number of Votes:", min_value=5, max_value=150, value=30, step=5)
        
        pref_years = st.slider("Release Era Range:", min_value=1930, max_value=2024, value=(1990, 2024))
        
        pref_count = st.slider("Number of Results:", min_value=3, max_value=25, value=9)

    with col_b:
        filtered_movies = engine.recommend_by_preferences(
            selected_genres=pref_genres,
            min_rating=pref_rating,
            min_votes=pref_votes,
            year_range=pref_years,
            top_n=pref_count
        )
        
        st.markdown(f"#### Found {len(filtered_movies)} Top-Ranked Movies Matching Criteria")
        
        if filtered_movies.empty:
            st.warning("No movies matched all criteria. Try lowering the minimum votes or rating threshold!")
        else:
            for _, row in filtered_movies.iterrows():
                genre_badges = "".join([f"<span class='badge'>{g}</span>" for g in row['genres'].split('|')])
                st.markdown(f"""
                <div class="movie-card" style="margin-bottom: 10px;">
                    <div style="display: flex; justify-content: space-between;">
                        <h4 style="margin: 0; color: #f8fafc;">{row['title']}</h4>
                        <span style="color: #f59e0b; font-weight: bold;">⭐ {row['vote_average']} / 5.0</span>
                    </div>
                    <div style="margin: 4px 0 8px 0; color: #94a3b8; font-size: 0.85rem;">
                        Year: <b>{int(row['year']) if pd.notnull(row['year']) else 'N/A'}</b> | 
                        Total Votes: <b>{row['vote_count']}</b> | 
                        IMDb Weighted Score: <b>{row['weighted_rating']}</b>
                    </div>
                    <div>{genre_badges}</div>
                </div>
                """, unsafe_allow_html=True)

# ----------------- TAB 3: EXPLORATORY DATA ANALYSIS -----------------
with tab3:
    st.subheader("📊 Exploratory Data Analysis (EDA) on MovieLens")
    st.markdown("Visualizing rating behaviors, genre frequencies, and historical release distributions.")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown("##### 1. Distribution of User Ratings")
        fig1, ax1 = plt.subplots(figsize=(7, 4.2))
        sns.histplot(ratings['rating'], bins=10, kde=True, color='#FF4B4B', ax=ax1)
        ax1.set_xlabel("Rating Value (0.5 to 5.0)")
        ax1.set_ylabel("Rating Count")
        ax1.set_title("User Rating Distribution (Median: 3.5)", fontsize=11)
        st.pyplot(fig1)
        
    with chart_col2:
        st.markdown("##### 2. Top 10 Most Rated Movies of All Time")
        top_voted = movies_df.sort_values(by='vote_count', ascending=False).head(10)
        fig2, ax2 = plt.subplots(figsize=(7, 4.2))
        sns.barplot(data=top_voted, x='vote_count', y='clean_title', hue='clean_title', palette='viridis', legend=False, ax=ax2)
        ax2.set_xlabel("Number of Ratings")
        ax2.set_ylabel("")
        ax2.set_title("Most Reviewed Movies in MovieLens", fontsize=11)
        st.pyplot(fig2)

    chart_col3, chart_col4 = st.columns(2)
    
    with chart_col3:
        st.markdown("##### 3. Movies Count per Genre")
        all_genres_series = movies_df['genres'].str.split('|').explode()
        all_genres_series = all_genres_series[all_genres_series != '(no genres listed)']
        genre_counts = all_genres_series.value_counts().head(12)
        
        fig3, ax3 = plt.subplots(figsize=(7, 4.2))
        sns.barplot(x=genre_counts.values, y=genre_counts.index, hue=genre_counts.index, palette='magma', legend=False, ax=ax3)
        ax3.set_xlabel("Number of Movies")
        ax3.set_ylabel("")
        ax3.set_title("Top 12 Genre Distributions", fontsize=11)
        st.pyplot(fig3)

    with chart_col4:
        st.markdown("##### 4. Movie Release Volume Over Time")
        valid_years = movies_df[movies_df['year'].between(1930, 2020)]
        fig4, ax4 = plt.subplots(figsize=(7, 4.2))
        sns.histplot(valid_years['year'], bins=30, color='#3b82f6', ax=ax4, kde=True)
        ax4.set_xlabel("Release Year")
        ax4.set_ylabel("Number of Movies Released")
        ax4.set_title("Release Year Trend (1930 - 2020)", fontsize=11)
        st.pyplot(fig4)

# ----------------- TAB 4: ALGORITHMS & ARCHITECTURE -----------------
with tab4:
    st.subheader("🧠 Mathematical Foundations & Recommendation Architecture")
    
    st.markdown("""
    #### 1. Content-Based Filtering (NLP TF-IDF & Cosine Similarity)
    - **Mechanism:** Builds a metadata soup combining genre keywords and user-generated folksonomy tags.
    - **TF-IDF Vectorization:** Transforms tokenized textual features into a term-frequency inverse-document-frequency matrix.
    - **Cosine Similarity:**
    $$Sim_{content}(A, B) = \\frac{\\mathbf{A} \\cdot \\mathbf{B}}{\\|\\mathbf{A}\\| \\|\\mathbf{B}\\|}$$
    
    #### 2. Collaborative Filtering (Matrix Factorization via TruncatedSVD)
    - **Mechanism:** Decomposes the high-dimensional user-item rating matrix $R \\in \\mathbb{R}^{M \\times N}$ into low-rank latent factor spaces:
    $$R \\approx U \\Sigma V^T$$
    - **Offline Validation:** Evaluated on an 80/20 train-test split.
    - **RMSE:** ~2.84 | **MAE:** ~2.63 on unnormalized sparse test split.
    
    #### 3. Bayesian Weighted Rating (IMDb Formulation)
    Prevents movies with few 5-star ratings from outranking masterpieces with thousands of votes:
    $$WR = \\left(\\frac{v}{v + m}\\right) \\cdot R + \\left(\\frac{m}{v + m}\\right) \\cdot C$$
    where $v$ is vote count, $m$ is the 70th percentile threshold, $R$ is average rating, and $C$ is the mean rating across all movies.
    
    #### 4. Hybrid Recommendation Engine
    Fuses Content-Based and Collaborative Filtering using a linear convex combination with Bayesian popularity prior:
    $$Score_{Hybrid} = \\alpha \\cdot Sim_{content} + (1 - \\alpha) \\cdot Sim_{collab} + \\beta \\cdot Normalized(WR)$$
    """)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #64748b; font-size: 0.85rem;'>ElevateLabs AIML Internship Project | Designed & Developed with Python, Scikit-Learn & Streamlit</p>", unsafe_allow_html=True)
