import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Zara AI Churn Prediction",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM STYLING (Zara-inspired)
# ==========================================
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #fcfcfc;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        color: #000000;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Metrics */
    div[data-testid="stMetricValue"] {
        color: #C41E3A; /* Zara Red */
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #f4f4f4;
        border-right: 1px solid #e0e0e0;
    }
    
    /* Success & Error Text */
    .success-text { color: #28a745; font-weight: bold; font-size: 1.2rem; }
    .danger-text { color: #C41E3A; font-weight: bold; font-size: 1.2rem; }
</style>
""", unsafe_allow_html=True)


# ==========================================
# DATA & MODEL LOADING
# ==========================================
@st.cache_data
def load_data():
    data_path = 'zara_customers.csv'
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return None

@st.cache_resource
def train_model(df):
    if df is None:
        return None, None
    
    # Preprocessing identical to original script
    X = df.drop(columns=['customer_id', 'churn'])
    y = df['churn']
    
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    # Cap outliers
    for c in ['monetary_eur', 'avg_order_value', 'recency_days']:
        lo, hi = X[c].quantile(0.01), X[c].quantile(0.99)
        X[c] = X[c].clip(lo, hi)
        
    preproc = ColumnTransformer([
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median'))]), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
        ]), categorical_features)
    ])
    
    # Advanced Model: Gradient Boosting
    pos_weight = (y == 0).sum() / (y == 1).sum()
    sample_weights = np.where(y == 1, pos_weight, 1.0)
    
    model = Pipeline([
        ('prep', preproc),
        ('clf', GradientBoostingClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.06,
            subsample=0.8, min_samples_leaf=15,
            max_features='sqrt', random_state=42
        ))
    ])
    
    model.fit(X, y, clf__sample_weight=sample_weights)
    return model, X.columns.tolist()

df = load_data()
model, feature_cols = train_model(df)


# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/f/fd/Zara_Logo.svg", width=150)
st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Select View:", [
    "Overview & Metrics",
    "Exploratory Data Analysis",
    "AI Churn Predictor",
    "Data Explorer"
])

st.sidebar.markdown("---")
st.sidebar.markdown("**Antigravity AI Dashboard**")
st.sidebar.caption("v1.0.0 | Premium Edition")


# ==========================================
# PAGE CONTENT
# ==========================================

if menu == "Overview & Metrics":
    st.title("ZARA AI Customer Churn Dashboard")
    st.markdown("Monitor customer retention metrics, view analytical reports, and predict churn risk in real-time using advanced Machine Learning.")
    
    if df is not None:
        st.markdown("### Top-Level Metrics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Customers", f"{len(df):,}")
        with col2:
            churn_rate = df['churn'].mean() * 100
            st.metric("Overall Churn Rate", f"{churn_rate:.1f}%")
        with col3:
            avg_ltv = df['monetary_eur'].mean()
            st.metric("Average LTV", f"€{avg_ltv:.2f}")
        with col4:
            retention_rate = 100 - churn_rate
            st.metric("Retention Rate", f"{retention_rate:.1f}%")
            
        st.markdown("---")
        st.markdown("### Executive Summary")
        st.info("The AI engine indicates that **Recency**, **Frequency**, and **Return Rates** are the primary drivers of churn. Customers exhibiting an interaction with high return rates and low frequency represent the most significant flight risk.")
        
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            if os.path.exists("01_eda_churn_segments.png"):
                st.image("01_eda_churn_segments.png", caption="Churn Distribution Segments")
            else:
                st.info("💡 Churn Segment plot not found. Run 'zara_churn_analysis.py' to generate.")
        with col_img2:
            if os.path.exists("06_model_comparison.png"):
                st.image("06_model_comparison.png", caption="Model Performance Metrics")
            else:
                st.info("💡 Model Comparison plot not found. Run 'zara_churn_analysis.py' to generate.")
    else:
        st.error("Dataset not found. Please ensure 'zara_customers.csv' is in the directory.")

elif menu == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis (EDA)")
    st.markdown("Visual insights extracted from our synthetic retail dataset.")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Distributions", "Correlations", "Model Eval", "Feature Importance"])
    
    with tab1:
        st.subheader("RFM Feature Distributions")
        if os.path.exists("02_eda_rfm_distributions.png"):
            st.image("02_eda_rfm_distributions.png", use_container_width=True)
            
    with tab2:
        st.subheader("Correlation Heatmap")
        if os.path.exists("03_eda_correlation_heatmap.png"):
            st.image("03_eda_correlation_heatmap.png", use_container_width=True)
            
    with tab3:
        st.subheader("Model Evaluation & Comparison")
        if os.path.exists("model_comparison.csv"):
            st.markdown("**Performance Metrics Table**")
            df_comp = pd.read_csv("model_comparison.csv")
            st.dataframe(df_comp, use_container_width=True, hide_index=True)
        if os.path.exists("04_model_evaluation.png"):
            st.image("04_model_evaluation.png", use_container_width=True)
            
    with tab4:
        st.subheader("Top Churn Predictors (Feature Importance)")
        if os.path.exists("feature_importance.csv"):
            st.markdown("**Feature Importance Values**")
            df_feat = pd.read_csv("feature_importance.csv")
            st.dataframe(df_feat, use_container_width=True, hide_index=True)
        if os.path.exists("05_feature_importance.png"):
            st.image("05_feature_importance.png", use_container_width=True)

elif menu == "AI Churn Predictor":
    st.title("Real-Time AI Churn Predictor")
    st.markdown("Enter customer details to evaluate their probability of churning.")
    
    with st.form("prediction_form"):
        st.subheader("Customer Profile")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", min_value=16, max_value=100, value=30)
            gender = st.selectbox("Gender", ["Female", "Male"])
            country = st.selectbox("Country", ['Spain', 'USA', 'UK', 'France', 'Germany', 'Italy', 'Mexico', 'Other'])
            city_tier = st.selectbox("City Tier", ['Tier 1', 'Tier 2', 'Tier 3'])
            channel = st.selectbox("Shopping Channel", ['Online', 'Store', 'Mixed'])
            
        with col2:
            tenure_months = st.number_input("Tenure (Months)", min_value=1, max_value=120, value=12)
            recency_days = st.number_input("Recency (Days since last purchase)", min_value=0, max_value=500, value=30)
            frequency = st.number_input("Frequency (Total orders)", min_value=1, max_value=100, value=5)
            monetary_eur = st.number_input("Total Spend (€)", min_value=1.0, max_value=10000.0, value=300.0)
            avg_order_value = monetary_eur / frequency if frequency > 0 else 0
            st.markdown(f"**Auto-Calculated AOV:** €{avg_order_value:.2f}")

        with col3:
            app_sessions_30d = st.slider("App Sessions (30 Days)", 0, 50, 5)
            pages_per_session = st.slider("Avg Pages per Session", 1.0, 50.0, 5.0)
            wishlist_items = st.slider("Items in Wishlist", 0, 50, 3)
            email_open_rate = st.slider("Email Open Rate", 0.0, 1.0, 0.5)
            return_rate = st.slider("Return Rate", 0.0, 1.0, 0.2)
            discount_usage_ratio = st.slider("Discount Usage Ratio", 0.0, 1.0, 0.1)
            category_diversity = st.slider("Category Diversity", 1, 10, 3)
            preferred_category = st.selectbox("Preferred Category", ['Women', 'Men', 'Kids', 'Home', 'TRF', 'Beauty'])
            complained = st.selectbox("Filed Complaint?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")

        submit_button = st.form_submit_button("Predict Churn Risk", use_container_width=True)
        
    if submit_button:
        if model is None:
            st.error("Model is not initialized. Please ensure the dataset exists to train the model.")
        else:
            # Prepare input data
            input_dict = {
                'age': age,
                'gender': gender,
                'country': country,
                'city_tier': city_tier,
                'tenure_months': tenure_months,
                'recency_days': recency_days,
                'frequency': frequency,
                'monetary_eur': monetary_eur,
                'avg_order_value': avg_order_value,
                'app_sessions_30d': app_sessions_30d,
                'pages_per_session': pages_per_session,
                'wishlist_items': wishlist_items,
                'email_open_rate': email_open_rate,
                'category_diversity': category_diversity,
                'preferred_category': preferred_category,
                'channel': channel,
                'return_rate': return_rate,
                'discount_usage_ratio': discount_usage_ratio,
                'complained': complained
            }
            
            input_df = pd.DataFrame([input_dict])
            
            # Ensure columns match training data
            for col in feature_cols:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[feature_cols]
            
            # Predict
            prob = model.predict_proba(input_df)[0][1]
            pred = int(prob > 0.5)
            
            st.markdown("---")
            st.subheader("Prediction Results")
            
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.metric("Churn Probability", f"{prob*100:.1f}%")
                
            with res_col2:
                if pred == 1:
                    st.markdown("<p class='danger-text'>⚠️ HIGH RISK: Customer is likely to churn!</p>", unsafe_allow_html=True)
                else:
                    st.markdown("<p class='success-text'>✅ SAFE: Customer is likely to be retained.</p>", unsafe_allow_html=True)
                    
            st.progress(prob)
            
            # Antigravity Easter Egg
            if prob < 0.05:
                st.balloons()
                st.success("🌟 AntiGravity Event: This customer's loyalty is defying gravity! (0-5% churn risk)")

elif menu == "Data Explorer":
    st.title("Raw Data Explorer")
    st.markdown("Browse and filter the underlying Zara customer dataset.")
    
    if df is not None:
        # Filtering options
        st.sidebar.markdown("---")
        st.sidebar.subheader("Filters")
        churn_filter = st.sidebar.multiselect("Churn Status", [0, 1], default=[0, 1], format_func=lambda x: "Churned" if x==1 else "Retained")
        country_filter = st.sidebar.multiselect("Country", df['country'].unique(), default=df['country'].unique())
        
        filtered_df = df[df['churn'].isin(churn_filter) & df['country'].isin(country_filter)]
        
        st.dataframe(filtered_df, use_container_width=True)
        
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Filtered Data (CSV)", data=csv, file_name="zara_filtered_data.csv", mime="text/csv")
    else:
        st.error("Dataset not found. Please ensure 'zara_customers.csv' is in the directory.")
