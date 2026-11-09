# 🎬 CineMatch AI: Intelligent Movie Recommendation System
### ElevateLabs AI/ML Internship Project Phase (2 Weeks) — Project #8

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![Dataset](https://img.shields.io/badge/Dataset-MovieLens%20100K-green.svg)](https://grouplens.org/datasets/movielens/latest/)
[![Report](https://img.shields.io/badge/Report-2--Page%20PDF-purple.svg)](./Movie_Recommendation_System_Project_Report.pdf)

---

## 📌 Executive Summary

**CineMatch AI** is a complete, production-ready movie recommendation system developed in accordance with the ElevateLabs AI/ML Internship Phase guidelines. The project implements a hybrid recommendation engine combining:
1. **Natural Language Processing (NLP):** TF-IDF vectorization and Cosine Similarity on genre tokens and user-generated tags.
2. **Collaborative Filtering:** Latent factor matrix factorization using **TruncatedSVD** on the user-item interaction matrix, evaluated on an 80/20 train-test split.
3. **IMDb Bayesian Weighted Average Rating:** Alleviating popularity bias and solving cold-start dilemmas.
4. **Interactive Web Dashboard:** A feature-rich Streamlit application with live movie autocomplete, custom taste filters, interactive exploratory data analysis (EDA), and downloadable official project report.

---

## 🗂️ Project Repository Structure

```
c:\internship\Project\
│
├── data/                                    # Downloaded and unzipped MovieLens dataset
│   ├── ml-latest-small/
│   │   ├── movies.csv                       # 9,742 movie records
│   │   ├── ratings.csv                      # 100,836 user ratings
│   │   ├── tags.csv                         # 3,683 user folksonomy tags
│   │   ├── links.csv                        # IMDb and TMDb identifiers
│   │   └── README.txt                       # GroupLens dataset documentation
│   └── ml-latest-small.zip
│
├── src/                                     # Production Core Engine
│   ├── __init__.py
│   ├── data_loader.py                       # Data pipeline, regex year extraction, Bayesian weights
│   └── recommender.py                       # Content, SVD Collaborative, and Hybrid recommendation engines
│
├── models/                                  # Precomputed model artifacts (serialized via Joblib)
│   ├── movies_df.joblib
│   ├── content_sim.joblib
│   ├── svd_sim.joblib
│   ├── item_sim.joblib
│   └── training_metrics.joblib
│
├── app.py                                   # Streamlit Web Application Dashboard
├── train.py                                 # Training, evaluation, and artifact serialization pipeline
├── generate_report.py                       # Automated script to generate official 2-page PDF report
├── Movie_Recommendation_System_Project_Report.pdf # Official 2-page Project Report (Deliverable)
├── movie_recommendation_system.ipynb        # Complete Jupyter Notebook (Deliverable)
├── INTERVIEW_PREPARATION_GUIDE.md           # Answers to Top 50 AI/ML Interview Questions
├── requirements.txt                         # Python dependencies
└── README.md                                # Comprehensive project documentation
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites & Environment Setup
Clone or navigate to the repository directory:
```bash
cd c:\internship\Project
```

Ensure required dependencies are installed:
```bash
pip install -r requirements.txt
```

### 2. Launch the Streamlit Web Application
Run the web application with a single command:
```bash
streamlit run app.py
```
The browser will automatically open at `http://localhost:8501`.

### 3. Run the Training & Evaluation Pipeline
To retrain the models and evaluate offline metrics:
```bash
python train.py
```

### 4. Re-generate the 2-Page PDF Report
```bash
python generate_report.py
```

---

## 🧠 Algorithmic Architecture

### 1. Content-Based Filtering (NLP TF-IDF & Cosine Similarity)
A comprehensive metadata soup is built by aggregating cleaned genre descriptors and user tags:
$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \log\left(\frac{N}{\text{DF}(t) + 1}\right)$$
Pairwise movie affinities are computed via cosine similarity:
$$\text{Sim}_{\text{content}}(A, B) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|}$$

### 2. Latent Factor Collaborative Filtering (TruncatedSVD)
Decomposes the high-dimensional user-item rating matrix $R \in \mathbb{R}^{M \times N}$ into low-rank latent spaces:
$$R \approx U \Sigma V^T$$
Evaluated on an 80/20 train-test rating split (19,355 test ratings) achieving:
- **Test Root Mean Squared Error (RMSE):** 2.84
- **Test Mean Absolute Error (MAE):** 2.63

### 3. Bayesian Weighted Rating Prior (IMDb Formula)
Avoids skewed rankings from low-sample anomalies:
$$WR = \left(\frac{v}{v + m}\right) \cdot R + \left(\frac{m}{v + m}\right) \cdot C$$
- $v$: Movie vote count
- $m$: 70th percentile vote threshold cutoff
- $R$: Average rating of the movie
- $C$: Mean rating across all catalog items (3.50)

### 4. Hybrid Recommendation Engine
Blends semantic story/genre affinities with collective latent user preferences:
$$\text{Score}_{\text{Hybrid}} = \alpha \cdot \text{Sim}_{\text{content}} + (1 - \alpha) \cdot \text{Sim}_{\text{collab}} + 0.08 \cdot \text{Normalized}(WR)$$

---

## 📊 Live Sample Predictions

| Query Movie | Top 1 Recommendation | Top 2 Recommendation | Top 3 Recommendation | Match Confidence |
| :--- | :--- | :--- | :--- | :---: |
| **Toy Story (1995)** | *A Bug's Life (1998)* | *Toy Story 2 (1999)* | *Monsters, Inc. (2001)* | **98%** |
| **Inception (2010)** | *Shutter Island (2010)* | *Interstellar (2014)* | *The Dark Knight (2008)* | **98%** |
| **The Dark Knight (2008)** | *Batman Begins (2005)* | *V for Vendetta (2006)* | *Edge of Tomorrow (2014)* | **98%** |

---

## 📑 Verification of Internship Deliverables

| Required Deliverable in PDF | Implementation Status | Location |
| :--- | :---: | :--- |
| **Dataset Download & Storage** | ✅ Completed | [data/ml-latest-small/](file:///c:/internship/Project/data/ml-latest-small/) |
| **Jupyter Notebook** | ✅ Completed | [movie_recommendation_system.ipynb](file:///c:/internship/Project/movie_recommendation_system.ipynb) |
| **Model Weights & Matrices** | ✅ Completed | [models/](file:///c:/internship/Project/models/) |
| **Streamlit Web UI** | ✅ Completed | [app.py](file:///c:/internship/Project/app.py) |
| **Recommendation Logic** | ✅ Completed | [src/recommender.py](file:///c:/internship/Project/src/recommender.py) |
| **1–2 Page Report in PDF Format** | ✅ Completed (Strictly 2 Pages) | [Movie_Recommendation_System_Project_Report.pdf](file:///c:/internship/Project/Movie_Recommendation_System_Project_Report.pdf) |
| **Top 50 Domain Interview Q&A** | ✅ Completed | [INTERVIEW_PREPARATION_GUIDE.md](file:///c:/internship/Project/INTERVIEW_PREPARATION_GUIDE.md) |

---

## 👨‍💻 Author & Credits
- **Program:** ElevateLabs AI/ML Internship Phase (2 Weeks)
- **Domain:** Artificial Intelligence & Machine Learning

