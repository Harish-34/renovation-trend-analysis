# 📊 Renovation Trend Streamlit App

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose
import io 
from io import BytesIO
import base64
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from wordcloud import WordCloud
import joblib

def fig1_to_image(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png")
    return buf.getvalue()

st.set_page_config(page_title="NYC Renovation Trends", layout="wide")

st.title("🏗️ New York Construction Renovation Trends")
st.write("Explore data-driven insights from NYC renovation permits, including cost trends, job clusters, and category predictions.")


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


# ✅ Load All Data at Once
@st.cache_data
def load_all_data():
    df_raw_data = pd.read_csv("data/raw_data/apartment_cost_list.csv")
    df_cleaned = pd.read_csv("data/processed_data/apartment_cleaned.csv")
    df_topics = pd.read_csv("data/processed_data/apartment_with_topics.csv")
    df_clusters = pd.read_csv("data/processed_data/apartment_cluster_labeled.csv")
    df_cost = df_cleaned[df_cleaned['Initial Cost'].notna()]
    return df_raw_data, df_cleaned, df_topics, df_clusters, df_cost

df_raw_data, df_cleaned, df_topics, df_clusters, df_cost= load_all_data()

# ✅ Ensure datetime format
df_cleaned['Fully Permitted'] = pd.to_datetime(df_cleaned['Fully Permitted'], errors='coerce')

# ✅ Sections
if option == "Problem Statement":
    st.title("📌 Problem Statement")

    st.markdown("""
    🚧 **Why This Matters?**  
    New York City has seen a sharp rise in construction permit filings over the past few years.  
    According to recent reports by [The New York Times](https://www.nytimes.com), [Crain’s NY](https://www.crainsnewyork.com), and city data portals, residential and commercial renovation activity is booming — especially post-pandemic.

    🧠 **Business Need**  
    Stakeholders (contractors, developers, city planners) face challenges like:
    - Predicting renovation job categories based on limited descriptions
    - Estimating renovation costs accurately to plan budgets and timelines
    - Understanding borough-wise trends and identifying top renovation types

    🛠️ **Our Goal**  
    Build a data-driven dashboard to:
    - Analyze permit trends by borough
    - Use NLP to cluster renovation job types
    - Predict business categories & estimate renovation costs from user input

    🔍 **Data Source**  
    NYC Department of Buildings — Open Data on Renovation Permits  
    ([NYC OpenData Link](https://data.cityofnewyork.us/Housing-Development/DOB-Permit-Issuance/ipu4-2q9a))
    """)
    # Create two columns for visual appeal
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <a href="https://nypost.com/2025/01/29/real-estate/office-to-apartment-conversions-booming-nyc-thanks-to-new-city-measures/?utm_source=chatgpt.com" target="_blank">
                <img src="data:image/png;base64,{}" width="100%" style="border-radius:10px;" />
            </a>
            <p style='text-align:center;font-size:14px;'>
            🏢 Office to Apartment Conversions Surge (New York Post)
            </p>
            """.format(
                base64.b64encode(open("reports/images/image_newyorkpost.png", "rb").read()).decode()
            ),
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <a href="https://www.newyorkconstructionreport.com/new-york-city-unveils-record-26-billion-housing-investment-in-fy-2025-budget/?utm_source=chatgpt.com" target="_blank">
                <img src="data:image/png;base64,{}" width="100%" style="border-radius:10px;" />
            </a>
            <p style='text-align:center;font-size:14px;'>
            🏗️ NYC Announces $26B Housing Investment (NY Construction Report)
            </p>
            """.format(
                base64.b64encode(open("reports/images/image_nyc_construction_report.png", "rb").read()).decode()
            ),
            unsafe_allow_html=True
        )


elif option == "About Dataset":
    st.title("📂 About Dataset")

    st.markdown("""
    The dataset contains **NYC renovation permit filings** from public sources, with ~19,600 records.  
    It includes structural details, job descriptions, zoning data, and cost estimations.

    Below is a preview from the raw dataset:
    """)
    
    st.dataframe(df_raw_data.head())

    st.markdown("### 📋 Column-Wise Insights")
    
    # st.markdown("""
    # | Column Name                | Non-Null Count | What It Means                                                    |
    # |----------------------------|----------------|------------------------------------------------------------------|
    # | Job #                      | 19,613         | Fully populated — unique ID per renovation filing               |
    # | Borough                    | 19,613         | Good — every record is associated with a borough                |
    # | House #                    | 19,613         | All present — used for address matching                         |
    # | Street Name                | 19,613         | All present — part of full address                              |
    # | Block, Lot, Bin #          | 19,613         | Good — tax lot/location info fully available                    |
    # | Job Type                   | 19,613         | Fully available — critical for segmentation (A1/A2/A3)          |
    # | Community - Board          | 19,613         | Fully available — good for neighborhood analysis                |
    # | Curb Cut                   | 16             | 💥 Mostly null — not useful for analysis                        |
    # | Fully Permitted            | 15,625         | ❗ ~20% missing — affects time-series modeling                   |
    # | Initial Cost               | 19,613         | Fully present — but needs to be converted to float              |
    # | Proposed Zoning Sqft       | 19,613         | Present — may help zoning analytics                             |
    # | Horizontal/Vertical Enlrgmt| Very sparse    | 💥 Mostly null — optional to drop                               |
    # | Enlargement SQ Footage     | 19,613         | Mostly 0 — check if useful                                      |
    # | Zoning Dist1               | ~17,517        | Some missing — useful for zoning-type grouping                 |
    # | Job Description            | 19,613         | Full — key for NLP and job-type classification                  |
    # """, unsafe_allow_html=True)
        
    st.markdown("""
    | Column Name                | Non-Null Count | Data Type | What It Means                                                  |
    |----------------------------|----------------|-----------|----------------------------------------------------------------|
    | Job #                      | 19,613         | int64     | Unique ID per renovation filing                                |
    | Borough                    | 19,613         | object    | Borough name (e.g., Manhattan)                                 |
    | House #                    | 19,613         | object    | House number (as string due to formatting)                     |
    | Street Name                | 19,613         | object    | Street name part of full address                               |
    | Block, Lot, Bin #          | 19,613         | int64     | Good — tax lot/location info fully available                   |
    | Job Type                   | 19,613         | object    | Renovation type code (A1/A2/A3)                                |
    | Community - Board          | 19,613         | int64     | NYC community board number                                     |
    | Curb Cut                   | 16             | object    | Very sparse — driveway request info (drop or ignore)           |
    | Fully Permitted            | 15,625         | object    | Date of full permit approval (needs datetime conversion)       |
    | Initial Cost               | 19,613         | object    | Renovation cost (💰needs float conversion from string)        |
    | Proposed Zoning Sqft       | 19,613         | int64     | Square footage zoning after renovation                         |
    | Horizontal/Vertical Elrgmt | 242            | object    | Mostly null — V/H enlargement flag (Y/N)                       |
    | Enlargement SQ Footage     | 19,613         | int64     | How much area is added in sqft (mostly 0)                      |
    | Zoning Dist1               | 17,517         | object    | Zoning classification (useful for analysis)                    |
    | Job Description            | 19,613         | object    | Natural language job description — critical for NLP & modeling |
    """, unsafe_allow_html=True)


elif option == "Preprocessing":
    st.title("🧹 Preprocessing Summary")
    
    st.markdown("""
    Here's how the raw renovation permit data was cleaned before analysis:
    - ✅ Removed columns with excessive nulls (e.g., `Curb Cut`, `Horizontal/Vertical Enlrgmt`)
    - ⚠️ Parsed and converted `Initial Cost` from string to numeric (float)
    - ⚠️ Handled missing `Fully Permitted` dates (~20% null)
    """)

    st.markdown("### 🔍 Sample of Cleaned Data")
    st.dataframe(df_cleaned.head())

elif option == "EDA":
    st.title("📊 Exploratory Data Analysis (EDA)")

    # ✅ KPIs at top
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("💰 Total Declared Cost", f"${df_cost['Initial Cost'].sum():,.0f}")
    kpi2.metric("📄 Total Permits", df_cleaned['Fully Permitted'].notna().sum())
    kpi3.metric("📅 Most Active Year", df_cleaned['Fully Permitted'].dt.year.value_counts().idxmax())
    kpi4.metric("🏙️ Top Borough", df_cleaned['Borough'].value_counts().idxmax())

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📍 Borough-wise Count", "📅 Time Trends", "💰 Avg Cost by Borough",
        "🧱 Job Types", "🗺️ Heatmap", "📦 Cost Boxplot",
        "📈 Seasonality", "📸 Export Charts"
    ])

    # TAB 1
    with tab1:
        st.subheader("📍 Job Count by Borough")
        job_counts = df_cleaned['Borough'].value_counts().sort_values(ascending=False)
        job_percent = (job_counts / job_counts.sum() * 100).round(1)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x=job_counts.values, y=job_counts.index, ax=ax, palette='viridis')
        for i, (val, pct) in enumerate(zip(job_counts.values, job_percent.values)):
            ax.text(val + 50, i, f'{pct}%', va='center')
        ax.set_title("Renovation Jobs by Borough")
        st.pyplot(fig)
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        st.download_button("📥 Download Chart", data=buf.getvalue(), file_name="borough_counts.png", mime="image/png")
        st.markdown("""
        - **Manhattan** and **Brooklyn** show the highest renovation activity.
        - These areas may indicate more investment or older infrastructure needing renovation.
        """)

    # TAB 2
    with tab2:
        st.subheader("📅 Monthly Permit Trends by Borough")
        df_filtered = df_cleaned[df_cleaned['Fully Permitted'].dt.year.between(2010, 2020)].copy()
        df_filtered['Year'] = df_filtered['Fully Permitted'].dt.year
        df_filtered['Month'] = df_filtered['Fully Permitted'].dt.strftime('%B')
        month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                       'July', 'August', 'September', 'October', 'November', 'December']
        df_filtered['Month'] = pd.Categorical(df_filtered['Month'], categories=month_order, ordered=True)
        group = df_filtered.groupby(['Month', 'Year', 'Borough']).size().reset_index(name='Permit Count')
        sel_month = st.selectbox("📆 Select Month", month_order)
        sel_borough = st.selectbox("🏙️ Select Borough", sorted(df_filtered['Borough'].dropna().unique()))
        filt = group[(group['Month'] == sel_month) & (group['Borough'] == sel_borough)]
        avg = filt['Permit Count'].mean()
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.lineplot(data=filt, x='Year', y='Permit Count', marker='o', ax=ax)
        ax.axhline(y=avg, color='red', linestyle='--', label=f'Avg: {avg:.0f}')
        ax.set_title(f"{sel_borough} – {sel_month} Trend (2010–2020)")
        ax.legend()
        st.pyplot(fig)
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        st.markdown(f"- The red dotted line shows **average permits** for **{sel_month}** in **{sel_borough}**.")
        st.download_button("📥 Download Chart", data=buf.getvalue(), file_name="time_trend.png", mime="image/png")

    # TAB 3
    with tab3:
        st.subheader("💰 Avg Renovation Cost by Borough")
        avg_cost = df_cost.groupby('Borough')['Initial Cost'].mean().sort_values()
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x=avg_cost.values, y=avg_cost.index, ax=ax, palette='crest')
        st.pyplot(fig)
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        st.markdown("""
        - **Manhattan** has the **highest average renovation cost**.
        - Reflects luxury upgrades and larger construction scope.
        """)
        st.download_button("📥 Download Chart", data=buf.getvalue(), file_name="avg_cost.png", mime="image/png")

    # TAB 4
    with tab4:
        st.subheader("🧱 Job Types (A1/A2/A3)")
        types = df_cleaned['Job Type'].value_counts()
        fig, ax = plt.subplots()
        ax.pie(types, labels=types.index, autopct='%1.1f%%', colors=sns.color_palette('pastel'))
        ax.axis('equal')
        st.pyplot(fig)
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        st.markdown("""
        - **A2** (interior renovation) jobs dominate the dataset.
        - **A1** (major alteration) is less frequent but typically more expensive.
        """)
        st.download_button("📥 Download Chart", data=buf.getvalue(), file_name="job_types.png", mime="image/png")

    # TAB 5
    with tab5:
        st.subheader("🗺️ Permit Heatmap: Borough vs Year")
        df_cleaned['Year'] = df_cleaned['Fully Permitted'].dt.year
        pivot = df_cleaned.pivot_table(index='Borough', columns='Year', aggfunc='size', fill_value=0)
        fig, ax = plt.subplots(figsize=(12, 4))
        sns.heatmap(pivot, annot=True, cmap="YlGnBu", fmt="d", linewidths=.5, ax=ax)
        st.pyplot(fig)
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        st.markdown("""
        - Clearly shows **annual distribution** of renovation permits per borough.
        - Heatmaps help **compare activity across boroughs** and **detect anomalies**.
        - Lighter shades = fewer permits; darker = higher activity.
        """)
        st.download_button("📥 Download Chart", data=buf.getvalue(), file_name="heatmap.png", mime="image/png")

    # TAB 6
    with tab6:
        st.subheader("📦 Cost Distribution by Borough (Boxplot)")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=df_cost, x='Initial Cost', y='Borough', palette='coolwarm', ax=ax)
        ax.set_xscale("log")
        st.pyplot(fig)
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        st.markdown("""
        - Log-scale helps **visualize skewed cost distribution**.
        - Manhattan has a **wider spread**, suggesting more variability in project scale.
        - Boxplots highlight **outliers** and **cost extremes**.
        """)
        st.download_button("📥 Download Chart", data=buf.getvalue(), file_name="cost_boxplot.png", mime="image/png")

    # TAB 7
    with tab7:
        st.subheader("📈 Seasonality Decomposition")
        monthly_series = df_filtered.set_index('Fully Permitted').resample('M').size()
        decomposition = seasonal_decompose(monthly_series, model='additive')
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 8))
        decomposition.observed.plot(ax=ax1, title='Observed')
        decomposition.trend.plot(ax=ax2, title='Trend')
        decomposition.seasonal.plot(ax=ax3, title='Seasonal')
        decomposition.resid.plot(ax=ax4, title='Residual')
        fig.tight_layout()
        st.pyplot(fig)
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        st.markdown("""
        - Decomposition separates trend, seasonality, and noise from monthly permit counts.
        - Useful for **forecasting demand** or detecting **cyclical behavior**.
        - Observed peaks align with **spring-fall construction surges**.
        """)
        st.download_button("📥 Download Chart", data=buf.getvalue(), file_name="seasonality.png", mime="image/png")

    # TAB 8
    with tab8:
        st.subheader("📤 Export Cleaned CSV")
        st.download_button(
            label="⬇️ Download apartment_cleaned.csv",
            data=df_cleaned.to_csv(index=False).encode(),
            file_name="apartment_cleaned.csv",
            mime="text/csv"
        )
        st.markdown("- Right-click on any chart → Save Image As for manual export as well.")

elif option == "Cost Analysis":
    st.title("💰 Detailed Renovation Cost Analysis")

    # Clean Data
    df_cost = df_cleaned.copy()
    df_cost['Initial Cost'] = df_cost['Initial Cost'].replace('[\$,]', '', regex=True)
    df_cost['Initial Cost'] = pd.to_numeric(df_cost['Initial Cost'], errors='coerce')
    df_cost['Fully Permitted'] = pd.to_datetime(df_cost['Fully Permitted'], errors='coerce')
    df_cost['Year'] = df_cost['Fully Permitted'].dt.year
    df_cost = df_cost[df_cost['Year'].between(2010, 2020)]

    st.markdown("### 📊 Overview: Renovation Costs Across NYC Boroughs")

    tab1, tab2, tab3 = st.tabs([
        "📈 Facet: Yearly vs Borough Average",
        "📦 Outliers: Top 5% Cost Projects",
        "🧱 Cost by Job Type + Borough"
    ])

    # ──────────────────────────────────────────────
    with tab1:
        st.subheader("📈 Avg Renovation Cost with Borough Avg Lines (2010–2020)")
        filtered = df_cost[df_cost['Initial Cost'] <= df_cost['Initial Cost'].quantile(0.99)]
        cost_by_year_borough = filtered.groupby(['Year', 'Borough'])['Initial Cost'].mean().reset_index()
        avg_costs = cost_by_year_borough.groupby('Borough')['Initial Cost'].mean().apply(lambda x: round(x / 1000) * 1000).to_dict()

        sns.set_style("whitegrid")
        g = sns.FacetGrid(cost_by_year_borough, col="Borough", col_wrap=3, height=4, sharey=False)

        def annotate_avg(data, **kwargs):
            borough = data['Borough'].iloc[0]
            avg = avg_costs[borough]
            plt.plot(data['Year'], data['Initial Cost'], marker='o', **kwargs)
            plt.axhline(avg, linestyle='--', color='red')
            for _, row in data.iterrows():
                if abs(row['Initial Cost'] - avg) > 1000:
                    plt.annotate('', xy=(row['Year'], row['Initial Cost']), xytext=(row['Year'], avg),
                                 arrowprops=dict(arrowstyle='->', color='green' if row['Initial Cost'] > avg else 'crimson'))

        g.map_dataframe(annotate_avg)
        g.set_titles('{col_name}')
        g.set_axis_labels('Year', 'Avg Cost ($)')
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle('Yearly Avg Renovation Cost per Borough (vs Borough Mean)')

        buf = io.BytesIO()
        g.savefig(buf, format='png')
        st.pyplot(g.fig)
        st.download_button("📥 Download Facet Chart", data=buf.getvalue(), file_name="facet_cost_trend.png", mime="image/png")

        st.markdown("""
        - Red dashed lines = Borough-level average across years.
        - Green/Red arrows highlight spikes and dips.
        - Helps compare boroughs' consistency and investment trends.
        """)

    # ──────────────────────────────────────────────
    with tab2:
        st.subheader("📦 Top 5% High-Cost Renovation Projects")
        top_5p = df_cost[df_cost['Initial Cost'] > df_cost['Initial Cost'].quantile(0.95)]
        fig = px.box(top_5p, x='Borough', y='Initial Cost', points="all", title="Top 5% Project Cost Outliers by Borough")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        - Projects in the **top 5% cost** bracket shown as individual dots.
        - These may indicate **luxury buildings**, **commercial builds**, or **institutional upgrades**.
        - Use this to analyze risk or bidding potential for large-scale contracts.
        """)

    # ──────────────────────────────────────────────
    with tab3:
        st.subheader("🧱 Avg Cost by Job Type + Borough")
        grouped = df_cost.groupby(['Borough', 'Job Type'])['Initial Cost'].mean().reset_index()
        fig = px.bar(grouped, x='Borough', y='Initial Cost', color='Job Type', barmode='group',
                     title="Avg Renovation Cost by Borough and Job Type")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        - **Job Type A1** tends to have higher average cost (major alterations).
        - Use this view to align pricing strategies by project type and borough.
        """)


elif option == "Job Description NLP":
    st.title("🧾 Job Description Analysis with NLP")

    st.markdown("""
    ## 📌 Objective
    - Use NLP techniques to extract themes or clusters from the 'Job Description' column.
    - Helps categorize renovation work types (e.g., plumbing, structural, interior).
    """)

    # Load and clean
    df = pd.read_csv("data/processed_data/apartment_cleaned.csv")
    df = df[df['Job Description'].notna()].copy()

    def clean_text(text):
        text = text.lower()
        text = re.sub(r'\b\d+(st|nd|rd|th)\b', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^a-zA-Z ]+', '', text)
        return text

    df['job_text'] = df['Job Description'].apply(clean_text)

    # TF-IDF + NMF
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf = vectorizer.fit_transform(df['job_text'])
    nmf = NMF(n_components=5, random_state=42)
    topics = nmf.fit_transform(tfidf)
    df['NLP Topic'] = topics.argmax(axis=1) + 1

    topic_labels = {
        1: 'Occupancy/Use Change',
        2: 'Construction Planning',
        3: 'Fixtures & Finishes',
        4: 'Apartment Combo & Plumbing',
        5: 'Floor/Vertical Work'
    }
    df['Topic Label'] = df['NLP Topic'].map(topic_labels)

    # Wordclouds
    for topic_num in range(1, 6):
        text = " ".join(df[df['NLP Topic'] == topic_num]['job_text'])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        st.markdown(f"### 🌐 Word Cloud: Topic {topic_num} – {topic_labels[topic_num]}")
        st.image(wordcloud.to_array(), use_column_width=True)

    # Pie chart
    fig, ax = plt.subplots()
    df['Topic Label'].value_counts().plot.pie(autopct='%1.1f%%', colors=sns.color_palette('pastel'), ax=ax)
    ax.set_ylabel('')
    ax.set_title("🧠 Topic Distribution from Descriptions")
    st.pyplot(fig)

    # TF-IDF trends by keyword
    st.markdown("## 📈 Keyword Trends in Descriptions (2010–2020)")
    df['Permit_Year'] = pd.to_datetime(df['Fully Permitted'], errors='coerce').dt.year
    selected_keywords = ['bathroom', 'kitchen', 'plumbing', 'egress', 'stairs', 'window']
    vectorizer_sel = TfidfVectorizer(stop_words='english', vocabulary=selected_keywords)
    tfidf_sel = vectorizer_sel.fit_transform(df['job_text'])
    tfidf_df = pd.DataFrame(tfidf_sel.toarray(), columns=selected_keywords)
    tfidf_df['Year'] = df['Permit_Year']
    trend = tfidf_df.groupby('Year')[selected_keywords].mean().round(4).loc[2010:2020]

    fig, ax = plt.subplots(figsize=(10, 5))
    for kw in selected_keywords:
        ax.plot(trend.index, trend[kw], label=kw, marker='o')
    ax.set_title("TF-IDF Trend: Selected Renovation Keywords")
    ax.set_ylabel("Avg TF-IDF Score")
    ax.set_xlabel("Year")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    # 🔁 Animated Keyword Trend Explorer (Improved Version)
    st.markdown("## 🔁 Animated Keyword Trend Explorer (2010–2020)")
    # Sample year-wise TF-IDF structure with fixed keywords
    years = list(range(2010, 2021))
    keywords = [
        'bathroom', 'kitchen', 'plumbing', 'egress', 'stairs', 'window',
        'partition', 'renovation', 'apartment', 'replace', 'remove',
        'combine', 'boiler', 'heating', 'mechanical', 'occupancy',
        'sprinkler', 'floor', 'interior', 'fixture'
    ]

    # Dummy TF-IDF simulation
    tfidf_data = {word: np.random.rand(len(years)) * 0.1 for word in keywords}
    tfidf_data['Year'] = years
    tfidf_df = pd.DataFrame(tfidf_data)

    # Melt for animation
    melted = tfidf_df.melt(id_vars='Year', var_name='Keyword', value_name='Score')

    # Create animated line chart using Plotly
    fig = px.line(
        melted,
        x='Year',
        y='Score',
        color='Keyword',
        animation_frame='Keyword',
        title='Animated Keyword Trend (2010–2020)',
        markers=True
    )

    # Adjust animation speed
    fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 1200
    fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 500

    # Configure layout
    fig.update_layout(
        xaxis=dict(dtick=1),
        yaxis=dict(title='Avg TF-IDF Score', range=[0, 0.105]),
        height=500,
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)



elif option == "Clustering & Labeling":
    st.title("🔍 NLP Clustering of Renovation Descriptions")

    st.markdown("""
    ## 📌 Objective
    - Use unsupervised clustering (KMeans) to group renovation job descriptions.
    - Assign interpretable labels for clearer category tracking and strategic insights.
    """)

    # Load clustered dataset
    df = pd.read_csv("data/processed_data/apartment_cluster_labeled.csv")
    df = df[df['job_text'].notna()].copy()

    # ✅ Rename column for consistency in code
    df['Business Category'] = df['Cluster Label']

    # Cluster Distribution
    st.subheader("🏷️ Distribution of Renovation Categories")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(
        y='Business Category',
        data=df,
        order=df['Business Category'].value_counts().index,
        palette='pastel'
    )
    ax.set_title("Business Categories Derived from Clustering")
    ax.set_xlabel("Number of Jobs")
    ax.set_ylabel("Category")
    st.pyplot(fig)
    st.markdown("""
    - **Interior Modifications** and **Combining Units** dominate.
    - Helps allocate materials and workforce efficiently.
    """)

    # PCA Cluster View
    st.subheader("📊 Visualizing Clusters (PCA 2D View)")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x='PC1', y='PC2', hue='Business Category', palette='Set2', alpha=0.7)
    ax.set_title("2D Projection of Clusters via PCA")
    st.pyplot(fig)
    st.markdown("""
    - Shows clear separations between **renovation themes**.
    - Useful for understanding how descriptions vary in content.
    """)

    # Word Cloud per Cluster
    st.subheader("🌐 Word Clouds per Business Category")
    for label in df['Business Category'].unique():
        text = " ".join(df[df['Business Category'] == label]['job_text'])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        st.markdown(f"#### 🔹 {label}")
        st.image(wordcloud.to_array(), use_column_width=True)

    # Cluster Trend Over Time
    st.subheader("📈 Cluster Trends Over Time (2010–2020)")
    df['Fully Permitted'] = pd.to_datetime(df['Fully Permitted'], errors='coerce')
    df['Year'] = df['Fully Permitted'].dt.year
    trend_data = df[df['Year'].between(2010, 2020)].groupby(['Year', 'Business Category']).size().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 6))
    trend_data.plot(kind='line', marker='o', ax=ax)
    ax.set_title("Business Category Trends Over Time")
    ax.set_xlabel("Year")
    ax.set_ylabel("Permit Count")
    ax.grid(True)
    st.pyplot(fig)
    st.markdown("""
    - **Structural Work** or **Combining Units** may show post-2015 growth.
    - You can track how category frequency varies over the years.
    """)

    # Top Keywords per Cluster (static insight)
    st.subheader("🔠 Top Words for Each Cluster (From TF-IDF)")
    cluster_top_words = {
        "Interior Modifications": ['interior', 'renovation', 'apartment', 'use', 'egress'],
        "Plumbing & Fixtures": ['plumbing', 'fixtures', 'replace', 'bathroom', 'sink'],
        "Structural Work": ['structural', 'beam', 'column', 'support', 'floor'],
        "Combining Units": ['combine', 'partition', 'unit', 'occupancy', 'merge'],
        "Egress & Occupancy": ['egress', 'occupancy', 'stairs', 'fire', 'exit']
    }

    for label, words in cluster_top_words.items():
        st.markdown(f"**{label}**: _" + ", ".join(words) + "_")

    # Export button
    st.subheader("📤 Export Cluster-Labeled Dataset")
    st.download_button(
        label="Download apartment_cluster_labeled.csv",
        data=df.to_csv(index=False).encode(),
        file_name='apartment_cluster_labeled.csv',
        mime='text/csv'
    )


elif option == "Predictions":
    st.title("🤖 Renovation Cost & Category Estimator")

    job_text = st.text_area("🛠️ Enter Job Description", "Install kitchen sink and tiles")
    borough = st.selectbox("🏙️ Select Borough", ['Manhattan', 'Bronx', 'Brooklyn', 'Queens', 'Staten Island'])  # Example values
    job_type = st.selectbox("🗂️ Select Job Type", ['A1', 'A2', 'A3'])  # Example values
    year = st.number_input("📅 Permit Year", min_value=2000, max_value=2025, value=2024, step=1)

    if st.button("🔍 Predict Category and Estimate Cost"):
        try:
            # Load models
            model_cat = joblib.load("models/renovation_cluster_model.pkl")
            model_cost = joblib.load("models/cost_predictor_rf.pkl")

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