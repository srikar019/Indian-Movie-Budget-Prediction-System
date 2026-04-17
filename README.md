# 🎬 Indian Movie Budget Prediction & Recommendation System

> **ML-Powered Budget Recommendation, Explainability & Economic Analysis for Indian Cinema**

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![ML](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green.svg)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-purple.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)

---

## 📋 Problem Statement

Indian cinema is a $2.5B+ industry spanning multiple regional markets. Movie budgets are influenced by a complex web of factors — star power, director track records, genre, production house scale, release timing, and market dynamics. This project builds:

- ✅ **Budget Recommendation System** — not just prediction, but a recommended budget *range* based on similar successful films
- ✅ **SHAP Explainability** — understand *why* a budget is predicted (e.g., "Ranveer Singh adds ₹50 Cr")
- ✅ **Inflation-Adjusted Analysis** — normalize budgets across decades using CPI data for fair comparison
- ✅ **ROI & Profitability Prediction** — dual prediction: budget + financial success probability
- ✅ **Cross-Industry Comparison** — Bollywood, Tollywood, Kollywood, Sandalwood & Mollywood
- ✅ **Error Analysis** — professional-grade diagnostics showing where the model fails and why
- ✅ **Interactive Dashboard** — 8-page Streamlit app with 30+ visualizations

---

## 🏗️ Project Structure

```
Indian-Movie-Budget-Prediction/
│
├── data/
│   ├── raw_data.csv               # 3,994 real Indian movies (TMDB + Wikipedia)
│   ├── cleaned_enriched_data.csv   # Deduplicated + inflation-adjusted + ROI
│   ├── model_comparison.csv        # 9 models compared
│   ├── cv_results.json             # 5-fold cross-validation results
│   ├── error_analysis.json         # Model error diagnostics
│   └── eda_insights.json           # Statistical analysis results
│
├── src/
│   ├── data_cleaning.py            # Deduplication + computed scores + inflation
│   ├── feature_engineering.py      # 30+ engineered features
│   ├── preprocessing.py            # Encoding + scaling pipeline
│   ├── train_model.py              # Advanced training (stacking + SHAP + CV)
│   ├── eda_analysis.py             # Statistical EDA with hypothesis tests
│   ├── visualizations.py           # Chart generation module
│   ├── generate_dataset.py         # Original data generator
│   ├── fetch_tmdb_movies.py        # TMDB API data fetcher
│   └── build_ultimate_dataset.py   # Dataset builder
│
├── models/
│   ├── best_model.pkl              # Best performing model (Stacking Ensemble)
│   ├── Stacking_Ensemble.pkl       # RF + XGBoost + LightGBM → Ridge meta-learner
│   ├── preprocessor.pkl            # Saved preprocessing pipeline
│   └── *.pkl                       # All 9 trained models
│
├── app/
│   └── app.py                      # 8-page Streamlit web application
│
├── static/                         # Generated visualization images
├── notebooks/                      # Jupyter notebooks for EDA
│
├── requirements.txt
└── README.md
```

---

## 🔥 What Makes This Project Unique

### 1. Budget Recommendation System (Not Just Prediction)
Instead of outputting a single number ("₹50 Cr"), the system provides:
```
💎 Recommended Budget Range: ₹40 — ₹60 Crores
   Based on 47 similar films in dataset
🎯 Point Estimate: ₹48.32 Crores
```

### 2. Real Computed Actor/Director Scores
Actor popularity and director success rates are **computed from actual box office performance**, not hardcoded:
```python
Actor Score = (Normalized Avg Box Office × 0.4) + (Hit Rate × 0.3) + 
              (Movie Count × 0.15) + (Max Box Office × 0.15)
```

### 3. Inflation-Adjusted Economic Analysis
Budgets are normalized to 2024 ₹ using CPI data:
- ₹10 Cr in 1990 → ₹91 Cr in 2024 terms
- Reveals TRUE budget growth vs. nominal growth

### 4. SHAP Explainability
Every prediction comes with a waterfall chart showing **which factors contributed what amount**:
```
Star Power (Ranveer Singh):  +₹50 Cr
VFX Level 5:                 +₹30 Cr  
Genre (Sci-Fi):              +₹20 Cr
Director (SS Rajamouli):     +₹40 Cr
```

### 5. Stacking Ensemble (Not Just "Pick Best Model")
Instead of choosing the best single model, we **stack** Random Forest + XGBoost + LightGBM with a Ridge meta-learner — achieving better generalization.

### 6. Professional Error Analysis
```
Model MAPE on high-budget (≥₹200 Cr) = 45.2% vs low-budget (<₹50 Cr) = 28.1%
⚠️ Model struggles with high-budget films (limited samples)
```

---

## 📊 Feature Engineering (30+ Features)

### Computed Features (High Impact)
| Feature | Description | Source |
|---------|-------------|--------|
| `actor_popularity_score` | Star power (1-10) | Computed from box office history |
| `director_success_rate` | Hit rate (0-1) | % of movies that were hits |
| `production_house_strength` | Studio brand (1-10) | Budget + hit rate of studio |
| `genre_complexity_index` | Cost intensity (1-6) | Drama=1, Action=5, Sci-Fi=6 |
| `language_market_factor` | Market size (0.25-1.0) | Hindi=1.0, Kannada=0.45 |
| `hype_score` | Pre-release buzz (0-1) | Actors + Director + Sequel + VFX |

### Interaction Features
| Feature | Formula |
|---------|---------|
| `star_x_genre_tier` | Star Power × Genre Cost |
| `hype_x_market` | Hype Score × Language Market Size |
| `vfx_x_screens` | VFX Level × Screen Count |
| `production_x_genre` | Production Scale × Genre Complexity |

### Temporal & Market Features
| Feature | Description |
|---------|-------------|
| `industry_growth_factor` | Compound growth since 2000 |
| `release_season_score` | Festival timing (1-5, Diwali=5) |
| `is_ott_era` | Released after 2018 |
| `is_post_pandemic` | Released after 2021 |
| `market_maturity` | Industry development metric |

---

## 🧠 Machine Learning Models

| Model | Type | Notes |
|-------|------|-------|
| Linear Regression | Baseline | Simple linear model |
| Ridge Regression | Regularized | L2 regularization |
| Lasso Regression | Regularized | L1 regularization |
| Decision Tree | Tree-based | Single decision tree |
| Random Forest | Ensemble | Bagging of 300 trees |
| Gradient Boosting | Ensemble | Sequential boosting (400 trees) |
| XGBoost | Ensemble | Extreme gradient boosting |
| LightGBM | Ensemble | Light gradient boosting |
| **Stacking Ensemble** 🏆 | Meta-Ensemble | RF + XGB + LightGBM → Ridge |

### Evaluation Methodology
- **5-Fold Cross-Validation** — ensures results aren't due to a lucky split
- **Adjusted R²** — penalizes model complexity
- **MAE / RMSE / MAPE** — comprehensive error metrics
- **Overfitting Gap** — Train R² vs Test R² comparison
- **Error Analysis by Budget Range** — identifies weak spots

---

## 📈 Statistical Analysis (EDA)

Professional-grade analysis with hypothesis tests:

| Analysis | Method | Insight Example |
|----------|--------|-----------------|
| Budget normality | Shapiro-Wilk test | "Budget is NOT normally distributed (p < 0.05)" |
| Genre → Budget | Kruskal-Wallis test | "Genre significantly affects budget (p < 0.001)" |
| Feature correlations | Spearman ρ | "num_screens has strongest correlation (ρ = 0.82)" |
| Actor impact | Comparison test | "Top 10 actors increase budget by ₹45 Cr" |
| Temporal trends | Linear regression | "Budget increases ₹3.2 Cr/year (R² = 0.78)" |
| Outlier detection | IQR method | "23 outliers detected (5.8%)" |

---

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Data Cleaning + Feature Engineering
```bash
python src/data_cleaning.py
```

### 3. Run EDA Analysis
```bash
python src/eda_analysis.py
```

### 4. Train Models (with Cross-Validation + SHAP + Error Analysis)
```bash
python src/train_model.py
```

### 5. Launch Web App
```bash
streamlit run app/app.py
```

---

## 🖥️ Web Application (8 Pages)

| Page | Features |
|------|----------|
| 🏠 **Dashboard** | Key metrics, trends, ROI distribution |
| 🔮 **Budget Recommender** | Range-based prediction, waterfall chart, factor breakdown |
| 📊 **Feature Importance** | Top 20 features, importance %, insight cards |
| 📈 **Trend Analysis** | Growth curves, inflation-adjusted, heatmaps, YoY |
| 🌍 **Industry Comparison** | Budget vs BO scatter, hit rates, ROI, sunburst |
| 🎭 **Genre & Actor Insights** | Genre deep-dive, top actors, director success scatter |
| 📉 **Error Analysis** | Model diagnostics, error by budget range, bias analysis |
| 🧠 **Model Performance** | 9-model comparison, CV results, overfitting check |

---

## ⚠️ Known Limitations & Future Work

### Known Limitations
- Model performs worse on very high-budget films (>₹200 Cr) due to limited samples
- Actor/Director scores are bootstrapped from the same dataset (circular dependency risk)
- Inflation index is approximate (CPI-IW based)

### Future Improvements
- [ ] Real-time TMDB API integration for live data
- [ ] Deep Learning with embeddings for actor/director representation
- [ ] NLP analysis of movie descriptions and trailer sentiment
- [ ] Network graph analysis of Director-Actor collaborations
- [ ] Conformal prediction for calibrated confidence intervals
- [ ] Time-series forecasting for industry-level budget predictions
- [ ] Deployment on Streamlit Cloud with CI/CD

---

## 📄 License

This project is open-source under the MIT License.

---

## 👤 Author

Advanced ML/Data Science project for analyzing Indian cinema economics.

**Tech Stack**: Python, Pandas, Scikit-Learn, XGBoost, LightGBM, SHAP, Streamlit, Plotly, SciPy
# Indian-Movie-Budget-Prediction-System
