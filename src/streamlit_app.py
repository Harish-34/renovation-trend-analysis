# 📊 Renovation Trend Streamlit App

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cost_estimator import CostEstimator
import seaborn as sns
import joblib

st.set_page_config(page_title="Renovation Trends", layout="wide")
st.title("🏗️ Renovation Trend Analysis Dashboard")
st.write("Welcome! Use the sidebar to explore different sections.")

# 🚀 Sidebar Navigation
with st.sidebar:
    st.title("📊 Navigation")
    option = st.radio("Go to", [
        "Problem Statement",
        "About Dataset",
        "Preprocessing",
        "EDA",
        "Cost Analysis",
        "Job Description NLP",
        "Clustering & Labeling",
        "Predictions"
    ])

st.write(f"### Currently Viewing: {option}")

# ✅ Load All Data at Once
@st.cache_data
def load_all_data():
    df_cleaned = pd.read_csv("../data/processed_data/apartment_cleaned.csv")
    df_topics = pd.read_csv("../data/processed_data/apartment_with_topics.csv")
    df_clusters = pd.read_csv("../data/processed_data/apartment_cluster_labeled.csv")
    return df_cleaned, df_topics, df_clusters

df_cleaned, df_topics, df_clusters = load_all_data()

# ✅ Sections
if option == "Problem Statement":
    st.title("📌 Problem Statement")
    st.markdown("""
    A construction firm wants to analyze NYC renovation permits to identify:
    - Where activity is high (by borough)
    - What type of work is common (NLP clustering)
    - Predict categories of new renovation jobs automatically
    """)

elif option == "About Dataset":
    st.title("📂 About Dataset")
    st.markdown("""
    - NYC renovation permits (~19K records)
    - Features: Borough, address, cost, description
    - Used for trend, text, and cluster analysis
    """)
    st.dataframe(df_cleaned.head())

elif option == "Preprocessing":
    st.title("🧹 Preprocessing Summary")
    st.markdown("""
    - Dropped high-null columns (e.g., Curb Cut)
    - Converted cost fields
    - Cleaned and saved as `apartment_cleaned.csv`
    """)
    st.dataframe(df_cleaned[['Borough', 'Initial Cost', 'Fully Permitted']].sample(5))

elif option == "EDA":
    st.title("📊 Borough-wise Permit Count")
    job_counts = df_cleaned['Borough'].value_counts()
    st.bar_chart(job_counts)

elif option == "Cost Analysis":
    st.title("💰 Average Renovation Cost by Borough")
    cost_df = df_cleaned[df_cleaned['Initial Cost'].notna()]
    cost_df['Initial Cost'] = cost_df['Initial Cost'].replace(r"[\$,]", "", regex=True).astype(float)
    borough_cost = cost_df.groupby('Borough')['Initial Cost'].mean().sort_values()
    fig, ax = plt.subplots()
    borough_cost.plot(kind='barh', ax=ax, color='teal')
    st.pyplot(fig)

elif option == "Job Description NLP":
    st.title("🧾 NLP Topics from Descriptions")
    if 'NLP Topic' in df_topics.columns:
        topic_counts = df_topics['NLP Topic'].value_counts().sort_index()
        st.bar_chart(topic_counts)
    else:
        st.warning("🚫 'NLP Topic' not found in the dataset.")

elif option == "Clustering & Labeling":
    st.title("🔍 Business Category Clustering")
    if 'Cluster Label' in df_clusters.columns:
        cluster_counts = df_clusters['Cluster Label'].value_counts()
        st.bar_chart(cluster_counts)
    else:
        st.warning("🚫 'Cluster Label' not found in the dataset.")

elif option == "Predictions":
    st.title("🤖 Renovation Cost & Category Estimator")

    job_text = st.text_area("🛠️ Enter Job Description", "Install kitchen sink and tiles")
    borough = st.selectbox("🏙️ Select Borough", ['Manhattan', 'Bronx', 'Brooklyn', 'Queens', 'Staten Island'])  # Example values
    job_type = st.selectbox("🗂️ Select Job Type", ['A1', 'A2', 'A3'])  # Example values
    year = st.number_input("📅 Permit Year", min_value=2000, max_value=2025, value=2024, step=1)

    if st.button("🔍 Predict Category and Estimate Cost"):
        try:
            # Load models
            model_cat = joblib.load("../models/renovation_cluster_model.pkl")
            model_cost = joblib.load("../models/cost_predictor_rf.pkl")

            # Predict cluster/category
            predicted_cluster = model_cat.predict([job_text])[0]

            # Estimate cost
            predicted_cost = model_cost.predict(
                job_text, borough, job_type, predicted_cluster, int(year)
            )

            st.success(f"🏷️ Predicted Category: **{predicted_cluster}**")
            st.info(f"💰 Estimated Renovation Cost: **${predicted_cost:,.0f}**")

        except Exception as e:
            st.error(f"⚠️ Prediction Failed: {e}")