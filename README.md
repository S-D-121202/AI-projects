# 📧 Spam Detection using TF-IDF and Gaussian Naive Bayes

A spam email classification project implemented **from scratch** using **TF-IDF** feature extraction and **Gaussian Naive Bayes**, without relying on pre-built machine learning classifiers.

---

## 🚀 Project Overview

This project classifies emails into **Spam** or **Ham (Non-Spam)** using classical **Natural Language Processing (NLP)** and **probabilistic machine learning** techniques.

The complete pipeline — preprocessing, feature extraction, training, and prediction — is implemented manually to gain a deep understanding of the underlying algorithms.

---

## 🧠 Methodology

### Text Preprocessing
- Lowercasing
- Punctuation and number removal
- Tokenization
- Stopword removal
- Lemmatization using POS tagging

### Feature Extraction
- Vocabulary built **only from training data**
- TF-IDF vectors of fixed dimensionality
- Unseen words in test data are safely ignored

### Classification
- Gaussian Naive Bayes
- Per-feature mean and variance estimation
- Log-probability computation for numerical stability

---

## 🛠️ Tech Stack

- **Python**
- **Pandas** – dataset handling
- **NumPy** – numerical computations
- **NLTK** – NLP preprocessing
- **SciPy** – Gaussian probability density (`norm.logpdf`)

