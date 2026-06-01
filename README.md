# 🛍️ Zara AI Churn Prediction & Analytics Dashboard

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

An end-to-end Machine Learning project designed to predict customer churn for Zara. This repository contains a synthetic dataset modeling fast-fashion consumer behavior, a comprehensive data analysis pipeline, and a premium interactive dashboard for real-time predictions.

---

## 🌟 Overview

Customer retention is critical in the fast-fashion industry. This project implements a robust AI-driven system to identify "at-risk" customers before they churn. By analyzing RFM (Recency, Frequency, Monetary) metrics, return rates, and digital engagement, the model provides actionable insights for marketing and retention strategies.

### Key Objectives:
- **Analyze** the drivers of customer churn in the fast-fashion sector.
- **Develop** a high-performance Gradient Boosting model for churn classification.
- **Visualize** findings through a Zara-inspired, professional dashboard.
- **Predict** churn risk for individual customers in real-time.

---

## 🚀 Features

- **Interactive Dashboard**: A multi-page Streamlit app with Zara's premium aesthetic.
- **Real-Time Predictor**: Input customer profile data and get an instant churn probability score.
- **Automated EDA**: Built-in scripts to generate distribution plots, correlation heatmaps, and churn segment analysis.
- **Model Comparison**: Benchmarking of Baseline Logistic Regression vs. Advanced Gradient Boosting.
- **Synthetic Data Engine**: A custom script to generate 10,000+ realistic customer records based on fashion-retail research.

---

## 🛠️ Tech Stack

- **Logic**: Python 3.8+
- **Machine Learning**: Scikit-Learn (Gradient Boosting, Logistic Regression)
- **Data Manipulation**: Pandas, NumPy
- **Visualizations**: Matplotlib, Seaborn
- **UI/UX**: Streamlit (with custom CSS injection)
- **Documentation**: Microsoft PowerPoint, LaTeX/PDF

---

## 📁 Project Structure

```text
├── app.py                      # Main Streamlit Dashboard
├── zara_churn_analysis.py      # Data generation & ML pipeline
├── zara_customers.csv          # Generated dataset (10,000 records)
├── requirements.txt            # Python dependencies
├── feature_importance.csv      # CSV output of model weights
├── model_comparison.csv        # Metrics for model comparison
├── 01_eda_churn_segments.png   # Visualization: Churn distribution
├── 02_eda_rfm_distributions.png# Visualization: RFM density plots
├── 03_eda_correlation_heatmap.png# Visualization: Feature correlations
├── 04_model_evaluation.png     # Visualization: ROC & PR curves
├── 05_feature_importance.png   # Visualization: Top churn drivers
├── 06_model_comparison.png     # Visualization: Performance metrics
├── Zara_Churn_Prediction_Research_Paper.pdf  # Detailed Research Document
└── Zara_Churn_Prediction_Presentation.pptx   # Executive Presentation
```

---

## ⚙️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/ghulammustafa603/AI-Driven-Churn-prediction-for-Zara-Asharib-khan-and-Meer-Abdullah-
cd AI-Driven-Churn-prediction-for-Zara
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate Data & Run Analysis (Optional)
If you want to regenerate the dataset and plots:
```bash
python zara_churn_analysis.py
```

### 4. Launch the Dashboard
```bash
streamlit run app.py
```

---

## 📊 Key Findings

- **Primary Churn Drivers**: Recency of purchase, Return Rates, and App Engagement.
- **Model Performance**: The Gradient Boosting model achieved significantly higher ROC-AUC compared to the baseline, effectively capturing non-linear behavior such as the "90-day dormancy cliff."
- **Strategic Insight**: Customers with high return rates (>40%) and low purchase frequency are the highest flight risks and should be targeted with personalized loyalty offers.

---

## 👥 Contributors

- **Asharib Khan**
- **Meer Abdullah**
