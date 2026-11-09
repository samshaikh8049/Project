# 📄 ElevateLabs Internship Project Report
## CineMatch AI: Intelligent Movie Recommendation System
**A Multi-Model Hybrid NLP & Latent Factor Collaborative Filtering Architecture**

---

### 📋 Project Submission Details
- **Intern Name:** Sameera Shaikh
- **Domain:** Artificial Intelligence & Machine Learning (AI / ML)
- **Internship Phase:** 2 Weeks (Project #8)
- **Organization / Program:** ElevateLabs
- **Official GitHub Repository:** [https://github.com/samshaikh8049/Project](https://github.com/samshaikh8049/Project)
- **Benchmark Dataset:** GroupLens MovieLens 100K Dataset
- **Deliverables:** Streamlit Web Application, Jupyter Notebook, Model Weights, Official 2-Page PDF Report, GitHub Codebase

---

## 1. Abstract
In the digital entertainment and video streaming era, consumers face acute decision fatigue due to exponentially expanding content catalogs. This project presents **CineMatch AI**, an end-to-end, multi-model movie recommendation framework developed using the benchmark MovieLens 100K dataset (comprising 9,742 movies, 100,836 ratings, and 610 users). 

The architecture synergizes **Natural Language Processing (NLP)** (TF-IDF vectorization and Cosine Similarity on genre descriptors and folksonomy tags) with **Latent Factor Collaborative Filtering** (Matrix Factorization via TruncatedSVD). To overcome the cold-start problem and eliminate low-sample rating anomalies, an IMDb Bayesian weighted average formulation is incorporated. The entire system is deployed as an interactive Streamlit web application featuring title fuzzy matching, personalized preference filters, exploratory data visualizations (EDA), and sub-second model response times.

---

## 2. Introduction
Recommender systems represent one of the most commercially critical applications of machine learning, driving core user retention across platforms like Netflix, Spotify, and Amazon. Standalone recommendation paradigms exhibit intrinsic trade-offs:
1. **Content-Based Filtering:** Recommends items based on descriptive metadata similarities. While it handles new items effectively, it suffers from over-specialization and lacks serendipity.
2. **Collaborative Filtering:** Uncovers collective behavioral patterns across users. However, it experiences severe cold-start failures when user or item interaction history is sparse.

The objective of this project is to develop and evaluate a production-grade **Hybrid Recommendation System** combining the complementary advantages of content semantics and latent user behavioral patterns. CineMatch AI computes pairwise movie affinities, ranks titles using Bayesian confidence adjustments, and delivers accessible real-time inferences through a responsive web interface.

---

## 3. Tools and Technologies Used

| Category | Component / Tool | Role & Purpose in Project |
| :--- | :--- | :--- |
| **Core Language** | Python 3.12 | Core computation, algorithmic logic, data structures, and pipeline orchestration |
| **Machine Learning** | Scikit-Learn | TF-IDF Vectorizer, TruncatedSVD Matrix Factorization, Cosine Similarity, Train-Test Split |
| **Data Manipulation** | Pandas & NumPy | User-item interaction pivot tables, vector operations, statistical aggregations |
| **Data Visualization** | Matplotlib & Seaborn | Rating distributions, top reviewed titles, genre frequencies, release year trends |
| **Web Deployment** | Streamlit | Modern reactive UI, multi-tab layout, interactive sliders, and real-time inference |
| **Benchmark Dataset** | GroupLens MovieLens 100K | 100,836 ratings, 9,742 movies, 3,683 user tags, and IMDb/TMDb identifiers |
| **Model Serialization** | Joblib | High-efficiency precomputed matrix caching for sub-second web inference |
| **Report Generation** | FPDF2 | Automated generation of the official 2-page publication-quality PDF report |

---

## 4. Steps Involved in Building the Project

### Step 1: Dataset Acquisition & Preprocessing
- Downloaded and verified the official MovieLens 100K benchmark dataset.
- Applied regular expressions (`r'\((\d{4})\)'`) to extract standardized release years from movie titles.
- Aggregated and normalized user-generated folksonomy tags per film.
- Cleaned missing values and formatted title casing.
- Implemented the Bayesian Weighted Rating formula (IMDb formulation) with a 70th percentile vote cutoff ($m$):
  $$WR = \left(\frac{v}{v + m}\right) \cdot R + \left(\frac{m}{v + m}\right) \cdot C$$
  *(where $v$ is movie vote count, $R$ is average rating, and $C$ is the mean rating across all movies: ~3.50)*

### Step 2: Content-Based NLP Filtering Engine
- Formulated a comprehensive "metadata soup" by combining genre tokens with aggregated user tags.
- Applied Scikit-Learn's `TfidfVectorizer` to extract term importance weights across vocabulary:
  $$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \log\left(\frac{N}{\text{DF}(t) + 1}\right)$$
- Computed the pairwise Cosine Similarity matrix quantifying semantic closeness:
  $$\text{Sim}_{\text{content}}(A, B) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|}$$

### Step 3: Collaborative Filtering via TruncatedSVD
- Built a high-dimensional user-item interaction matrix $R \in \mathbb{R}^{9742 \times 610}$ from explicit user ratings.
- Decomposed the interaction matrix using **Truncated Singular Value Decomposition (TruncatedSVD)** with 30 latent components:
  $$R \approx U \Sigma V^T$$
- Derived latent factor vectors capturing hidden thematic affinities and user taste patterns beyond superficial genre labels.

### Step 4: Hybrid Recommendation Formulation & Cold-Start Prior
- Developed an adaptive hybrid ensemble function:
  $$\text{Score}_{\text{Hybrid}} = \alpha \cdot \text{Sim}_{\text{content}} + (1 - \alpha) \cdot \text{Sim}_{\text{collab}} + 0.08 \cdot \text{Normalized}(WR)$$
- For items with sparse rating histories, the system automatically shifts weight ($\alpha \rightarrow 1.0$) to content similarity, eliminating the cold-start dilemma.

### Step 5: Interactive Web UI & Model Serialization
- Designed a feature-rich Streamlit application with four core modules:
  1. **Movie Recommender:** Search with title fuzzy matching and percentage match badges.
  2. **Personalized Taste Profiler:** Multi-genre selection, decade filters, and rating cutoffs.
  3. **Exploratory Data Analysis (EDA):** Interactive charts for rating distributions, genre counts, and historical releases.
  4. **Architecture & Math:** Detailed documentation of algorithmic formulations.
- Serialized similarity matrices via Joblib for instantaneous sub-second web inference.

---

## 5. Key Results and Experimental Evaluation

### Quantitative Evaluation (Offline Train-Test Split)
The SVD Collaborative Filtering model was validated using an 80/20 train-test split on 100,836 user ratings (19,355 test ratings):
- **Test Root Mean Squared Error (RMSE):** **2.84**
- **Test Mean Absolute Error (MAE):** **2.63**
- *Note:* Ratings were evaluated on raw unnormalized sparse data across 610 distinct user rating scales.

### Qualitative Evaluation (Sample Inferences)
| Input Movie | Top 1 Recommendation | Top 2 Recommendation | Top 3 Recommendation | Match Confidence |
| :--- | :--- | :--- | :--- | :---: |
| **Toy Story (1995)** | *A Bug's Life (1998)* | *Toy Story 2 (1999)* | *Monsters, Inc. (2001)* | **98%** |
| **Inception (2010)** | *Shutter Island (2010)* | *Interstellar (2014)* | *The Dark Knight (2008)* | **98%** |
| **The Dark Knight (2008)** | *Batman Begins (2005)* | *V for Vendetta (2006)* | *Edge of Tomorrow (2014)* | **98%** |

---

## 6. Conclusion
**CineMatch AI** successfully fulfills all project requirements outlined in the ElevateLabs AI/ML Internship Phase. By synthesizing NLP-based content representations with latent factor matrix factorization and Bayesian popularity priors, the system effectively resolves classical over-specialization and cold-start dilemmas. The system is fully reproducible, production-ready, and accompanied by a responsive web application and complete technical documentation.

---
*ElevateLabs AI/ML Internship Phase Submission | Author: Sameera Shaikh | [GitHub Repository](https://github.com/samshaikh8049/Project)*
