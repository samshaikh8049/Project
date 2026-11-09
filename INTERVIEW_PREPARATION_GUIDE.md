# ElevateLabs AI/ML Internship: Top 50 Interview Questions & Answers
### Complete Domain Master Guide & Project Defense Reference

---

## Section I: General AI & ML Concepts

### 1. What is the difference between AI, Machine Learning, and Deep Learning?
- **Artificial Intelligence (AI):** The broad discipline of creating machines capable of mimicking human cognitive functions (problem-solving, reasoning, learning).
- **Machine Learning (ML):** A subset of AI where systems automatically learn patterns from data and improve without explicit hand-coded rules.
- **Deep Learning (DL):** A subset of ML utilizing multi-layered artificial neural networks (ANNs) to automatically extract hierarchical representations from unstructured data (images, audio, text).
- *Project Context:* Our Movie Recommendation System utilizes ML (TruncatedSVD, Cosine Similarity) and NLP techniques (TF-IDF).

### 2. What are the main types of Machine Learning?
- **Supervised Learning:** Learning from labeled data $(X, y)$ to map inputs to outputs (e.g., Regression, Classification).
- **Unsupervised Learning:** Discovering hidden patterns or intrinsic structures in unlabeled data $(X)$ (e.g., Clustering, PCA, SVD Matrix Factorization).
- **Semi-Supervised Learning:** Leveraging small amounts of labeled data alongside large volumes of unlabeled data.
- **Reinforcement Learning (RL):** An agent learns optimal policy actions via environmental trial-and-error to maximize cumulative rewards.

### 3. Explain the difference between supervised and unsupervised learning.
- **Supervised:** Requires ground-truth target labels $y$. Evaluated via metrics like RMSE, Accuracy, F1-score. Examples: Linear Regression, SVM.
- **Unsupervised:** No target label $y$; algorithms explore data clustering, latent factors, or density estimation. Evaluated via Silhouette score, reconstruction error, or explained variance. Examples: K-Means, TruncatedSVD.

### 4. What is overfitting and underfitting?
- **Underfitting (High Bias):** The model is too simplistic to capture the underlying structure of the data, performing poorly on both training and test sets.
- **Overfitting (High Variance):** The model memorizes training noise rather than generalizing, yielding near-perfect training scores but poor generalization on unseen test data.
- *Remedies:* Cross-validation, regularization ($L_1, L_2$), pruning, dropout, gathering more data.

### 5. How do you evaluate a machine learning model?
- **Regression:** Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), $R^2$ score.
- **Classification:** Confusion Matrix, Accuracy, Precision, Recall, F1-Score, ROC-AUC curve.
- **Recommender Systems:** RMSE/MAE on test ratings, Precision@K, Recall@K, Mean Average Precision (MAP), Normalized Discounted Cumulative Gain (NDCG), and Catalog Coverage.

### 6. What is the bias-variance trade-off?
- Total Expected Error = $\text{Bias}^2 + \text{Variance} + \text{Irreducible Noise}$.
- **Bias:** Error introduced by approximating real-world complex problems with overly simple models.
- **Variance:** Sensitivity of the model to minor fluctuations in training data.
- *Goal:* Find the optimal model complexity that minimizes the sum of bias squared and variance.

### 7. Explain cross-validation and why it's important.
- Technique to evaluate model generalization ability by partitioning data into $K$ subsets. The model trains on $K-1$ folds and validates on the remaining fold, repeating $K$ times.
- *Importance:* Prevents data leakage, mitigates selection bias, and yields a reliable estimate of model performance on unseen data.

### 8. What are precision, recall, F1-score, and accuracy?
- **Accuracy:** $(TP + TN) / (TP + TN + FP + FN)$ — overall fraction of correct predictions. Misleading in imbalanced datasets.
- **Precision:** $TP / (TP + FP)$ — proportion of positive identifications that were genuinely positive. Critical when false positives are expensive (e.g., spam filter).
- **Recall (Sensitivity):** $TP / (TP + FN)$ — proportion of actual positives correctly identified. Critical when false negatives are costly (e.g., cancer detection).
- **F1-Score:** $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ — harmonic mean balancing precision and recall.

### 9. What is the difference between classification and regression?
- **Classification:** Predicts discrete categorical class labels (e.g., Fraud vs. Normal, Spam vs. Ham).
- **Regression:** Predicts continuous real-valued quantitative outputs (e.g., House prices, User movie rating from 0.5 to 5.0).

### 10. What are some real-world applications of AI/ML?
- Recommendation Engines (Netflix, YouTube, Spotify, Amazon)
- Natural Language Processing (ChatGPT, machine translation, voice assistants)
- Autonomous Driving & Computer Vision (Tesla, medical imaging diagnosis)
- Financial Fraud Detection & Algorithmic Trading

---

## Section II: Data Preprocessing & Feature Engineering

### 11. How do you handle missing data in a dataset?
- **Deletion:** Listwise or pairwise dropping (viable only if data is Missing Completely at Random (MCAR) and $<5\%$).
- **Imputation:** Mean/Median for numerical skewed features; Mode for categorical data.
- **Model-based Imputation:** KNN Imputer or Iterative MICE (Multivariate Imputation by Chained Equations).
- *Project Application:* In our MovieLens ratings matrix, unobserved entries are treated as missing user interactions and factorized via TruncatedSVD.

### 12. What is normalization and standardization?
- **Normalization (Min-Max Scaling):** Rescales data to a fixed range $[0, 1]$: $x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$. Sensitive to outliers.
- **Standardization (Z-score):** Centers data around $\mu=0$ with unit variance $\sigma=1$: $z = \frac{x - \mu}{\sigma}$. Retains outlier properties and aligns well with algorithms assuming Gaussian distributions.

### 13. What is one-hot encoding? When would you use it?
- Converting categorical variables into binary $(0/1)$ indicator columns.
- *Usage:* For nominal data without intrinsic ordinal ranking (e.g., Country, Genre). Avoid when cardinality is high ($>100$ unique categories) to prevent curse of dimensionality; target or frequency encoding is preferable then.

### 14. How do you handle categorical variables?
- Nominal (no order): One-Hot Encoding, Binary Encoding, Frequency/Target Encoding.
- Ordinal (ordered): Label/Ordinal Encoding (e.g., Low=1, Medium=2, High=3).
- High Cardinality Text: CountVectorizer, TF-IDF Vectorization, or entity embeddings.

### 15. What is feature selection and why is it important?
- The process of selecting a subset of relevant features for model construction.
- *Importance:* Reduces overfitting, improves model interpretability, shortens training latency, and mitigates the curse of dimensionality.
- *Methods:* Filter methods (Correlation, Mutual Info), Wrapper methods (RFE), Embedded methods (Lasso L1 penalty, Tree feature importances).

### 16. What is dimensionality reduction? Explain PCA.
- Reducing the number of random variables under consideration by obtaining principal variables.
- **Principal Component Analysis (PCA):** An unsupervised orthogonal linear transformation that maps data to a lower-dimensional subspace maximizing variance along principal eigenvectors of the covariance matrix.
- *Note:* In sparse user-item interaction matrices (like MovieLens), **TruncatedSVD** is used instead of PCA because it does not require centering the matrix, preserving memory sparsity.

### 17. What is the curse of dimensionality?
- As dimensional space increases, the volume grows exponentially, causing available data points to become extremely sparse. Distances between points converge to similar values, severely degrading Euclidean/distance-based algorithms (KNN, K-Means).

### 18. What is feature scaling and when should it be applied?
- Adjusting feature value ranges so gradient updates or distance calculations are not dominated by features with large magnitudes.
- *Apply to:* Distance-based algorithms (KNN, SVM, K-Means) and Gradient-Descent models (Linear/Logistic Regression, Neural Networks).
- *Not needed for:* Tree-based models (Decision Trees, Random Forest, XGBoost).

### 19. How do you deal with imbalanced datasets?
- **Resampling:** SMOTE (Synthetic Minority Over-sampling Technique), Random Under-Sampling.
- **Algorithmic:** Cost-sensitive learning (`class_weight='balanced'`), Focal Loss.
- **Evaluation Choice:** Use Precision-Recall AUC and F1-score rather than simple Accuracy.

### 20. Explain the role of EDA (Exploratory Data Analysis) in ML.
- Systematic exploration using summary statistics and data visualizations to discover patterns, detect anomalies/outliers, test hypotheses, and verify assumptions prior to model development.

---

## Section III: Machine Learning Algorithms

### 21. How does the Decision Tree algorithm work?
- Recursively splits feature space into rectangular regions using criteria like Gini Impurity (CART) or Information Gain (Entropy). At each node, the split that maximizes purity is selected until stopping criteria (max depth, min samples split) are reached.

### 22. What is the difference between bagging and boosting?
- **Bagging (Bootstrap Aggregation):** Trains multiple base learners in parallel on bootstrap samples and aggregates outputs via voting or averaging (e.g., Random Forest). Primarily reduces **variance**.
- **Boosting:** Trains base learners sequentially where each subsequent learner focuses on correcting residual errors of previous models (e.g., AdaBoost, XGBoost, LightGBM). Primarily reduces **bias**.

### 23. Explain how the K-Nearest Neighbors (KNN) algorithm works.
- Non-parametric, lazy learning algorithm that stores all training instances. To classify a query point, it calculates distance (Euclidean, Manhattan) to all training points, identifies the $K$ closest neighbors, and assigns the majority class (or average for regression).

### 24. What is the intuition behind Support Vector Machines (SVM)?
- Finds the optimal hyperplane that separates classes with the maximum geometric margin (distance between the hyperplane and closest data points, called support vectors). Employs Kernel Trick (RBF, Polynomial) to project non-linear data into higher-dimensional separable spaces.

### 25. How does the Naive Bayes classifier work?
- Probabilistic classifier rooted in Bayes' Theorem:
  $$P(C|X) = \frac{P(X|C) P(C)}{P(X)}$$
- Makes the "naive" conditional independence assumption that all features are mutually independent given the class label.

### 26. What is the difference between Random Forest and XGBoost?
- **Random Forest:** Parallel bagging ensemble of deep, unpruned decision trees with random feature subsampling. Resistant to overfitting.
- **XGBoost:** Sequential gradient boosting framework fitting shallow trees to the negative gradient of the loss function, incorporating explicit $L_1/L_2$ tree complexity regularization.

### 27. How do gradient descent and stochastic gradient descent differ?
- **Batch Gradient Descent:** Computes loss gradients across the entire dataset per step. Stable and deterministic, but computationally expensive on large datasets.
- **Stochastic Gradient Descent (SGD):** Updates weights using a single random sample per iteration. Fast and can escape local minima, but has noisy loss convergence.
- *Compromise:* Mini-batch Gradient Descent (e.g., batch size 32 or 64).

### 28. Explain logistic regression and where it’s used.
- Generalized linear model for binary classification. Computes a linear combination $z = w^T x + b$ and passes it through the Sigmoid activation function $\sigma(z) = \frac{1}{1 + e^{-z}}$ to yield an output probability between 0 and 1. Used in spam detection, click-through-rate prediction, and credit scoring.

### 29. What are ensemble models?
- Machine learning techniques that synthesize predictions from multiple base models to produce superior predictive performance compared to any individual constituent model (Bagging, Boosting, Stacking, Blending).

### 30. What is the difference between L1 and L2 regularization?
- **L1 Regularization (Lasso):** Adds penalty $\lambda \sum |w_i|$. Induces sparsity by driving non-informative feature weights strictly to zero, performing intrinsic feature selection.
- **L2 Regularization (Ridge):** Adds penalty $\lambda \sum w_i^2$. Shrinks weights uniformly toward zero without setting them strictly to zero, handling multicollinearity effectively.

---

## Section IV: Deep Learning & Neural Networks

### 31. What is a perceptron?
- The fundamental computational unit of artificial neural networks. Takes inputs $x_i$, weights $w_i$, bias $b$, computes linear sum $\sum w_i x_i + b$, and applies an activation function to generate output.

### 32. How do activation functions like ReLU, Sigmoid, and Tanh work?
- **Sigmoid:** $\sigma(z) = 1 / (1 + e^{-z})$, range $(0, 1)$. Vanishing gradients for large $|z|$.
- **Tanh:** $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$, zero-centered, range $(-1, 1)$.
- **ReLU:** $\max(0, z)$, computationally efficient, alleviates vanishing gradient for positive values; prone to "dying ReLU" (mitigated by Leaky ReLU).

### 33. What are epochs, batch size, and learning rate?
- **Epoch:** One complete forward and backward pass of the entire training dataset through the network.
- **Batch Size:** The number of training samples processed before the model parameters are updated.
- **Learning Rate:** Hyperparameter determining step size during gradient optimization.

### 34. What is the vanishing gradient problem?
- During backpropagation in deep networks, gradients calculated via the chain rule are repeatedly multiplied by small derivatives ($< 0.25$ in Sigmoid/Tanh). Gradients vanish asymptotically toward earlier layers, preventing weight updates.
- *Solutions:* ReLU activations, Batch Normalization, Residual Connections (ResNets).

### 35. What is the difference between CNN and RNN?
- **CNN (Convolutional Neural Network):** Employs spatial parameter-shared filters and pooling layers. Specialized for grid-structured spatial data (images, video).
- **RNN (Recurrent Neural Network):** Maintains a hidden state vector passed recurrently across time steps. Specialized for sequential temporal data (text, time series, audio).

### 36. How does an LSTM work and where is it used?
- Long Short-Term Memory network resolves RNN vanishing gradients via a memory cell state regulated by three non-linear gates: **Forget Gate** (what to discard), **Input Gate** (what new info to store), and **Output Gate** (what to output). Used in machine translation, speech recognition, and stock price forecasting.

### 37. What are convolutional layers in CNN?
- Layers that slide learnable 2D/3D kernel matrices over input feature maps, performing element-wise multiplications and summations to extract hierarchical spatial features (edges, textures, object motifs).

### 38. What is transfer learning?
- Taking a model pre-trained on a massive benchmark dataset (e.g., ImageNet, BERT on Wikipedia) and fine-tuning its weights on a specialized downstream target task with limited training samples.

### 39. What is dropout and why is it used?
- A regularization technique that randomly deactivates a fraction $p$ of neuron units during each training forward pass. Prevents complex co-adaptations between neurons, acting as an implicit ensemble.

### 40. How do you prevent overfitting in deep learning models?
- Early stopping, Dropout, Weight Decay ($L_2$), Batch Normalization, Data Augmentation, and reducing network capacity.

---

## Section V: Natural Language Processing (NLP)

### 41. What is tokenization in NLP?
- The process of segmenting continuous raw character sequences into discrete, meaningful tokens (words, subwords, or characters) suitable for numerical indexing (e.g., WordPiece, Byte-Pair Encoding).

### 42. How do word embeddings like Word2Vec or GloVe work?
- Distributed dense vector representations where semantically related words reside close to each other in latent geometric vector space.
  - **Word2Vec (Skip-gram / CBOW):** Predicts surrounding context words given center word (or vice-versa) using shallow neural networks.
  - **GloVe:** Factorizes global word-word co-occurrence matrix log-probabilities.

### 43. What is the difference between stemming and lemmatization?
- **Stemming:** Heuristic, rule-based chopping of word affixes, often yielding non-dictionary stems (e.g., "studies" $\rightarrow$ "studi"). Fast but coarse.
- **Lemmatization:** Morphological analysis using vocabulary and part-of-speech context, returning valid dictionary base forms (lemmas) (e.g., "better" $\rightarrow$ "good", "running" $\rightarrow$ "run").

### 44. What is TF-IDF and why is it used?
- **Term Frequency-Inverse Document Frequency:**
  $$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \log\left(\frac{N}{\text{DF}(t) + 1}\right)$$
- Quantifies importance of term $t$ relative to document $d$ within corpus $D$. Penalizes ubiquitous stop words ("the", "is") and amplifies domain-specific discriminative keywords.
- *Project Application:* Vectorizes our movie genres and metadata tags to compute content cosine similarity.

### 45. What are Transformers in NLP?
- Deep learning architecture relying entirely on **Self-Attention mechanisms** rather than recurrence. Enables parallel sequence processing and long-range dependency tracking. Foundations of BERT, GPT, and modern Large Language Models (LLMs).

---

## Section VI: Model Deployment & MLOps

### 46. How do you save and load a trained model in Python?
- **Joblib / Pickle:** Serializes Python object state to binary disk (`joblib.dump(model, 'model.joblib')`, `joblib.load('model.joblib')`). Optimal for Scikit-Learn matrices.
- **ONNX / TorchScript / SavedModel:** Cross-platform formats for deep neural networks.

### 47. What is model drift and how do you monitor it?
- **Concept Drift:** Statistical properties of the target concept change over time ($P(y|X)$ shifts).
- **Data Drift:** Distribution of input features changes ($P(X)$ shifts).
- *Monitoring:* Kolmogorov-Smirnov test, Population Stability Index (PSI), tracking online evaluation metrics (Click-Through-Rate, user retention).

### 48. How do you deploy a machine learning model as an API?
- Package preprocessing and model inference inside a web service framework (FastAPI, Flask) exposing RESTful JSON endpoints (e.g., `POST /predict`), containerized with Docker, and hosted on cloud infrastructure (AWS EC2, GCP, Azure).

### 49. What are common tools for deploying ML models (Flask, Docker, Streamlit)?
- **Streamlit:** Rapid interactive web UI and dashboarding framework for machine learning apps.
- **Flask / FastAPI:** Production API backends for high-throughput, low-latency microservices.
- **Docker:** OS-level virtualization packaging application code, dependencies, and system configurations into portable, reproducible container images.

### 50. Explain the CI/CD pipeline in MLOps.
- **Continuous Integration (CI):** Automated linting, unit testing, and validation of code and data pipelines upon git commit.
- **Continuous Delivery / Deployment (CD):** Automated container building, regression testing, and deployment to staging/production clusters.
- **Continuous Training (CT):** Automated retraining and evaluation triggers when model degradation or data drift is detected.

