# 🏗️ NYC Renovation Trend Analysis – Streamlit Dashboard

[![Streamlit App](https://img.shields.io/badge/Streamlit-View%20Live%20App-brightgreen?style=for-the-badge&logo=streamlit)](https://nyc9construction9rennovation9trend9analysis.streamlit.app/)

> An interactive Streamlit dashboard to explore renovation permit trends across NYC using **NLP**, **machine learning**, and **visual analytics**.

---

## 🚀 Live App

🔗 **Launch Dashboard:** [Click to open the Streamlit app](https://nyc9construction9rennovation9trend9analysis.streamlit.app/)

Experience real-time filtering, keyword exploration, topic modeling, clustering, and predictive analytics — all in one unified interface.

---

## 📖 Project Overview

This project delivers a comprehensive **NYC Renovation Trend Analysis System**, combining **NLP, clustering, machine learning**, and a **Streamlit-based interactive dashboard** for full-cycle analytics.

It utilizes historical renovation permit filings across NYC (2010–2020) to uncover patterns such as:

- 📊 **Borough-wise renovation activity** and cost trends  
- 🧠 **Topic modeling (TF-IDF + NMF)** to extract dominant renovation themes  
- 🔍 **KMeans clustering** with **PCA visualization** to classify renovation types  
- 📈 **Time-series decomposition** to detect seasonal patterns  
- 🤖 **ML-based predictions** of renovation category and estimated cost  
- 📸 **Exportable visuals** and cleaned datasets for further analysis

---

## 🧠 Key Features

The system integrates multiple analytical components and interactive tools to deliver actionable insights from raw permit data. Below are the major capabilities offered by the dashboard:

- 🔄 Interactive filtering by borough, job type, and time period  
- 📚 TF-IDF + NMF-based topic extraction from job descriptions  
- 📌 KMeans clustering with PCA-based 2D visualization  
- 🌐 Word clouds and labeled clusters for clear interpretation  
- 📅 Seasonal and yearly renovation trend analysis  
- 🧪 Live predictions using pre-trained ML models (category & cost)  
- 💾 Exportable charts and downloadable cleaned data

---

## 📂 Dataset Source

The dataset used in this project is sourced from the official NYC Open Data platform:

🔗 **[NYC DOB Permit Issuance Dataset](https://data.cityofnewyork.us/Housing-Development/DOB-Permit-Issuance/ipu4-2q9a)**

This dataset contains detailed information about construction permits issued by the **New York City Department of Buildings (DOB)** from 2010 to 2020. It includes attributes such as:

- Job type and description  
- Permit issuance date  
- Borough and location  
- Estimated cost of work  
- Permit type, status, and more

This rich dataset forms the foundation for all NLP, clustering, and cost prediction models used in the analysis.

---

## 📁 Project Structure

```bash
renovation-trend-analysis/
├── data/                         # Datasets used for analysis
│   ├── raw_data/                # Original downloaded NYC permit CSVs
│   └── processed_data/          # Cleaned/transformed datasets
│
├── models/                      # Trained ML model files
│   └── *.pkl                    # Saved models (e.g., NMF, KMeans, RandomForest)
│
├── reports/
│   └── images/                  # Exported plots, word clouds, visuals
│       ├── image1.png
│       ├── image2.png
│       └── ...
│
├── src/                         # Source code
│   └── streamlit_app.py         # Streamlit dashboard code
│
├── requirements.txt             # Python dependencies
└── README.md                    # Project overview and documentation
```
---

## 🔧 Tech Stack

This project brings together a full spectrum of data science and engineering tools to deliver interactive analytics, machine learning, and NLP in a single deployable app.

### 🎯 Frontend
- **Streamlit:** For building interactive web dashboards and visualization UI.
### 🧠 Machine Learning & Modeling
- **Scikit-learn:** Core ML library used for classification, regression, and clustering (Random Forest, KMeans, GridSearchCV).
- **Pipeline:** `Pipeline`, `make_pipeline` used for chaining preprocessing and modeling.
- **Model Persistence:** `joblib` for saving/loading trained ML models.
### 🗣️ Natural Language Processing (NLP)
- **TF-IDF (TfidfVectorizer):** Vectorizes job descriptions for topic modeling and ML.
- **NMF (Non-negative Matrix Factorization):** Topic extraction from TF-IDF features.
### 🔍 Clustering & Dimensionality Reduction
- **KMeans:** For job clustering based on textual features.
- **PCA (Principal Component Analysis):** For reducing dimensionality and visualizing clusters.
### 📈 Time Series & Seasonality
- **statsmodels:** `seasonal_decompose` used to identify seasonality and trends in renovation permits over time.
### 📊 Data Analysis & Preprocessing
- **Pandas, NumPy:** Core libraries for data cleaning, transformation, and manipulation.
- **Regex, IO, OS:** Utilities for data handling, parsing, and dynamic path management.
### 🖼️ Visualization
- **Matplotlib, Seaborn:** Static charts, trend lines, box plots.
- **Plotly Express:** Interactive bar and line charts in Streamlit.
- **WordCloud:** To generate word clouds for dominant job themes.

---

## 📸 Visualizations

Below are key visuals generated from the analysis — including permit trends, cost distributions, topic models, and clustering insights:

<table>
  <tr>
    <td><strong>📍 Renovation Jobs by Borough</strong><br><img src="reports/images/image1.png" width="100%"/></td>
    <td><strong>📅 Time Trends – Monthly Permit Analysis</strong><br><img src="reports/images/image2.png" width="100%"/></td>
  </tr>
  <tr>
    <td><strong>💰 Avg Initial Renovation Cost by Borough</strong><br><img src="reports/images/image3.png" width="100%"/></td>
    <td><strong>🧱 Distribution of Job Types (A1/A2/A3)</strong><br><img src="reports/images/image4.png" width="100%"/></td>
  </tr>
  <tr>
    <td><strong>📈 Cost Trends (2010–2020) by Borough</strong><br><img src="reports/images/image5.png" width="100%"/></td>
    <td><strong>📊 Faceted Cost Trend View (w/ Avg)</strong><br><img src="reports/images/image6.png" width="100%"/></td>
  </tr>
  <tr>
    <td><strong>🌐 Topic 1 Word Cloud</strong><br><img src="reports/images/image7.png" width="100%"/></td>
    <td><strong>🔍 PCA Cluster Plot</strong><br><img src="reports/images/image8.png" width="100%"/></td>
  </tr>
  <tr>
    <td><strong>📈 Cluster Trends Over Time</strong><br><img src="reports/images/image9.png" width="100%"/></td>
    <td><strong>🧩 Renovation Categories (Clustering)</strong><br><img src="reports/images/image10.png" width="100%"/></td>
  </tr>
</table>

---

## 🛠️ How to Run Locally

### ✅ Clone Repository

```bash
git clone https://github.com/Harish-34/renovation-trend-analysis.git
cd renovation-trend-analysis
```

### ✅ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
```

### ✅ Install Requirements

```bash
pip install -r requirements.txt
```

### ✅ Launch Streamlit App

```bash
streamlit run src/streamlit_app.py
```

---

## 📦 Deliverables

This project provides multiple actionable outputs that can be directly used or extended:

- 📥 **Downloadable Cleaned Datasets:** Ready-to-use CSV files for further analysis  
- 📊 **Exportable Visuals:** PNG charts for presentations and reports  
- 🤖 **Live Prediction Tool:** Instant ML-based predictions of renovation category and cost

---

## 🧾 Conclusion

The NYC Renovation Trend Analysis project combines NLP, clustering, time-series analysis, and machine learning into a single Streamlit dashboard. It enables both exploratory insights and predictive analytics on historical renovation permit data across NYC (2010–2020).

By integrating end-to-end data engineering with domain-specific visualizations and modeling, this solution showcases the power of real-world applied data science.

---

## 🙋‍♀️ Author

**Harish Chowdary**  
💼 [LinkedIn](https://www.linkedin.com/in/harish-chowdary)  
🌐 [Live App](https://nyc9construction9rennovation9trend9analysis.streamlit.app/)
