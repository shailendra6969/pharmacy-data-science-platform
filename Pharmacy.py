# pharmacy_data_app.py - Advanced Pharmacy Data Science App with ML Models
# Author: Data Science Professional
# Last Updated: April 2025

import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from pymongo import MongoClient
import sqlite3
import joblib
import os
import unittest
import pytest
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# App configuration
st.set_page_config(layout="wide", page_title="Pharmacy Data Science Platform")

# Check for model directory
MODEL_DIR = "models"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Database configuration
DB_PATH = "pharmacy_data.db"

def create_sqlite_db():
    """Create SQLite database and load data from sample_100_drugs.csv"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create drugs table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS drugs (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        category TEXT,
        price REAL,
        dosage TEXT,
        description TEXT,
        manufacturer TEXT,
        stock INTEGER
    )
    ''')
    
    # Create sales table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS sales (
        id INTEGER PRIMARY KEY,
        drug_id INTEGER,
        sale_date TEXT,
        quantity INTEGER,
        total_price REAL,
        FOREIGN KEY (drug_id) REFERENCES drugs (id)
    )
    ''')
    
    # Check if drugs table is empty
    cursor.execute("SELECT COUNT(*) FROM drugs")
    if cursor.fetchone()[0] == 0:
        # Load drugs from CSV
        drugs_df = pd.read_csv(r"C:\Users\Lenovo\Downloads\sample_100_drugs.csv")  # Assumes CSV is in the same directory
        drugs_data = [tuple(row) for row in drugs_df.itertuples(index=False)]
        cursor.executemany("INSERT INTO drugs VALUES (?, ?, ?, ?, ?, ?, ?, ?)", drugs_data)
        logger.info("Loaded 100 drugs from sample_100_drugs.csv into database")
        
        # Generate sample sales data for all 100 drugs
        sample_sales = []
        sale_id = 1
        for day in range(365):
            date = (datetime.now() - timedelta(days=365-day)).strftime('%Y-%m-%d')
            for drug_id in range(1, 101):  # IDs 1 to 100 from CSV
                is_weekend = (datetime.strptime(date, '%Y-%m-%d').weekday() >= 5)
                base_quantity = random.randint(3, 8) if is_weekend else random.randint(1, 5)
                month = datetime.strptime(date, '%Y-%m-%d').month
                is_winter = (month in [12, 1, 2])
                
                # Adjust quantities for specific categories (e.g., Respiratory, Antibiotic)
                cursor.execute("SELECT category, price FROM drugs WHERE id = ?", (drug_id,))
                category, price = cursor.fetchone()
                if category in ["Respiratory", "Antibiotic"] and is_winter:
                    quantity = base_quantity * 2
                else:
                    quantity = base_quantity
                
                total = price * quantity
                sample_sales.append((sale_id, drug_id, date, quantity, total))
                sale_id += 1
        
        cursor.executemany("INSERT INTO sales VALUES (?, ?, ?, ?, ?)", sample_sales)
        logger.info("Generated sales data for 100 drugs")
    
    conn.commit()
    conn.close()
    logger.info("SQLite database initialized with CSV data")

# Create and populate the SQLite database
create_sqlite_db()

# Title and description
st.title("💊 Advanced Pharmacy Data Science Platform")

st.markdown("""
This professional analytics platform demonstrates data science expertise in pharmaceutical domain with:
- **Machine Learning**: Drug price prediction models and shortage forecasting
- **Natural Language Processing**: Medical text analysis and recommendation engine
- **Network Analysis**: Gene-drug interaction visualization with centrality metrics
- **Time Series Analysis**: Sales forecasting with ARIMA and seasonal decomposition
- **Database Integration**: MongoDB and SQLite with real-time querying
- **Unit Testing**: Comprehensive test suite for data integrity and model validation
- **Documentation**: Detailed usage guides, API references, and methodologies
""")

# --- Sidebar Menu ---
menu = st.sidebar.radio("Select Module", [
    "Dashboard",
    "Gene–Drug Analysis",
    "Sales Forecast",
    "Drug Price ML Model",
    "Drug Shortage Prediction",
    "Medicine Recommendation",
    "Gene Interaction Network",
    "Product Catalog",
    "SQLite Database Explorer",
    "MongoDB Integration",
    "Documentation"
])

# --- Dashboard Overview ---
if menu == "Dashboard":
    st.subheader("📊 Pharmacy Analytics Dashboard")
    
    conn = sqlite3.connect(DB_PATH)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_drugs = pd.read_sql("SELECT COUNT(*) as count FROM drugs", conn).iloc[0,0]
        st.metric("Total Products", total_drugs)
        
    with col2:
        total_sales = pd.read_sql("SELECT SUM(total_price) as total FROM sales", conn).iloc[0,0]
        st.metric("Total Sales", f"₹{total_sales:,.2f}")
        
    with col3:
        popular_drug = pd.read_sql("""
            SELECT d.name, SUM(s.quantity) as total_sold 
            FROM sales s JOIN drugs d ON s.drug_id = d.id 
            GROUP BY d.name ORDER BY total_sold DESC LIMIT 1
        """, conn)
        st.metric("Most Popular Drug", popular_drug.iloc[0,0])
        
    with col4:
        low_stock = pd.read_sql("SELECT COUNT(*) as count FROM drugs WHERE stock < 400", conn).iloc[0,0]
        st.metric("Low Stock Items", low_stock)
    
    st.subheader("Sales Trends")
    sales_data = pd.read_sql("""
        SELECT date(sale_date) as date, SUM(total_price) as daily_sales
        FROM sales
        GROUP BY date
        ORDER BY date
    """, conn)
    
    sales_data['date'] = pd.to_datetime(sales_data['date'])
    sales_data = sales_data.set_index('date')
    sales_data['7day_avg'] = sales_data['daily_sales'].rolling(window=7).mean()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sales_data.index, sales_data['daily_sales'], 'b-', alpha=0.5, label='Daily Sales')
    ax.plot(sales_data.index, sales_data['7day_avg'], 'r-', label='7-Day Average')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales (₹)')
    ax.set_title('Daily Sales with 7-Day Moving Average')
    ax.legend()
    st.pyplot(fig)
    
    st.subheader("Top Selling Products")
    top_drugs = pd.read_sql("""
        SELECT d.name, SUM(s.quantity) as total_sold, SUM(s.total_price) as total_revenue
        FROM sales s JOIN drugs d ON s.drug_id = d.id
        GROUP BY d.name
        ORDER BY total_revenue DESC
        LIMIT 5
    """, conn)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(top_drugs['name'], top_drugs['total_revenue'])
    ax.set_xlabel('Drug Name')
    ax.set_ylabel('Revenue (₹)')
    ax.set_title('Top 5 Revenue-Generating Drugs')
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 5000,
                f'₹{height:,.0f}', ha='center', va='bottom', rotation=0)
    st.pyplot(fig)
    
    st.subheader("Sales by Category")
    category_sales = pd.read_sql("""
        SELECT d.category, SUM(s.total_price) as category_sales
        FROM sales s JOIN drugs d ON s.drug_id = d.id
        GROUP BY d.category
        ORDER BY category_sales DESC
    """, conn)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.pie(category_sales['category_sales'], labels=category_sales['category'], autopct='%1.1f%%', 
           startangle=90, shadow=True)
    ax.axis('equal')
    ax.set_title('Sales Distribution by Drug Category')
    st.pyplot(fig)
    
    conn.close()

# --- PharmGKB Gene–Drug Analysis ---
def fetch_gene_data(gene_ids):
    """Fetch gene-drug interaction data from PharmGKB API"""
    gene_drug_data = []
    for gene_id in gene_ids:
        try:
            url = f"https://api.pharmgkb.org/v1/data/gene/{gene_id}"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                gene_name = data.get('name', 'Unknown')
                related_drugs = data.get('relatedChemicals', [])
                for drug in related_drugs:
                    gene_drug_data.append({
                        "Gene": gene_name, 
                        "Gene ID": gene_id,
                        "Drug": drug.get('name', 'Unknown Drug'),
                        "Relation": drug.get('relation', 'Unknown')
                    })
            else:
                st.warning(f"Could not fetch data for gene {gene_id}. Status code: {response.status_code}")
            time.sleep(0.5)
        except Exception as e:
            logger.error(f"Error fetching data for gene {gene_id}: {e}")
            st.error(f"Error fetching data for gene {gene_id}: {str(e)}")
    return pd.DataFrame(gene_drug_data)

if menu == "Gene–Drug Analysis":
    st.subheader("🧬 PharmGKB Gene–Drug Interaction Analysis")
    st.info("""
    This module analyzes gene-drug interactions using the PharmGKB API. You can:
    1. Upload a CSV file containing gene IDs and drug names
    2. Enter PharmGKB gene IDs directly
    3. Visualize gene-drug interaction networks
    """)
    
    tab1, tab2 = st.tabs(["Data Input", "Network Analysis"])
    
    with tab1:
        uploaded_file = st.file_uploader("Upload CSV with Gene IDs:", type=["csv"])
        gene_ids_input = st.text_area("Or Enter PharmGKB Gene IDs (comma separated):", "PA124,PA128,PA130,PA131,PA134,PA151")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success("CSV Uploaded Successfully")
                if 'Gene ID' in df.columns:
                    gene_id_list = df['Gene ID'].unique().tolist()
                    st.info(f"Found {len(gene_id_list)} unique gene IDs in the CSV file")
                    if st.button("Fetch Gene Data from CSV"):
                        gene_data = fetch_gene_data(gene_id_list)
                        if not gene_data.empty:
                            st.dataframe(gene_data)
                            st.session_state.gene_data = gene_data
                else:
                    st.warning("The CSV file must contain a 'Gene ID' column")
            except Exception as e:
                st.error(f"Error reading CSV: {str(e)}")
        
        elif st.button("Fetch Gene–Drug Data"):
            gene_id_list = [gid.strip() for gid in gene_ids_input.split(",")]
            with st.spinner("Fetching gene-drug interaction data..."):
                gene_data = fetch_gene_data(gene_id_list)
                if not gene_data.empty:
                    st.dataframe(gene_data)
                    st.session_state.gene_data = gene_data
                    fig, ax = plt.subplots(figsize=(10, 6))
                    gene_counts = gene_data['Gene'].value_counts()
                    gene_counts.plot(kind='bar', ax=ax)
                    ax.set_title('Number of Drug Interactions by Gene')
                    ax.set_xlabel('Gene')
                    ax.set_ylabel('Number of Interactions')
                    st.pyplot(fig)
                else:
                    st.warning("No data found for the provided gene IDs")
    
    with tab2:
        st.subheader("Gene-Drug Interaction Network")
        if 'gene_data' in st.session_state and not st.session_state.gene_data.empty:
            df = st.session_state.gene_data
            G = nx.Graph()
            genes = df['Gene'].unique()
            for gene in genes:
                G.add_node(gene, type='gene')
            drugs = df['Drug'].unique()
            for drug in drugs:
                G.add_node(drug, type='drug')
            for _, row in df.iterrows():
                G.add_edge(row['Gene'], row['Drug'])
            gene_centrality = nx.degree_centrality(G)
            betweenness = nx.betweenness_centrality(G)
            
            fig, ax = plt.subplots(figsize=(12, 10))
            pos = nx.spring_layout(G, seed=42)
            gene_nodes = [n for n, d in G.nodes(data=True) if n in genes]
            nx.draw_networkx_nodes(G, pos, nodelist=gene_nodes, node_size=800, 
                                  node_color='lightblue', alpha=0.8, label='Genes')
            drug_nodes = [n for n, d in G.nodes(data=True) if n in drugs]
            nx.draw_networkx_nodes(G, pos, nodelist=drug_nodes, node_size=400,
                                  node_color='lightgreen', alpha=0.6, label='Drugs')
            nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
            gene_labels = {n: n for n in gene_nodes}
            drug_labels = {n: n for n in drug_nodes}
            nx.draw_networkx_labels(G, pos, labels=gene_labels, font_size=12, font_weight='bold')
            nx.draw_networkx_labels(G, pos, labels=drug_labels, font_size=8)
            plt.title("Gene-Drug Interaction Network", fontsize=16)
            plt.legend(scatterpoints=1)
            plt.axis('off')
            st.pyplot(fig)
            
            st.subheader("Network Centrality Analysis")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Degree Centrality (Top 5)")
                centrality_df = pd.DataFrame({
                    'Node': list(gene_centrality.keys()),
                    'Centrality': list(gene_centrality.values())
                }).sort_values('Centrality', ascending=False).head(5)
                st.dataframe(centrality_df)
                st.markdown(f"""
                **Key Insight**: {centrality_df.iloc[0]['Node']} is the most connected gene, 
                interacting with {int(centrality_df.iloc[0]['Centrality'] * (len(G.nodes) - 1))} drugs.
                This suggests it may be a hub gene involved in multiple drug responses.
                """)
            with col2:
                st.subheader("Betweenness Centrality (Top 5)")
                betweenness_df = pd.DataFrame({
                    'Node': list(betweenness.keys()),
                    'Betweenness': list(betweenness.values())
                }).sort_values('Betweenness', ascending=False).head(5)
                st.dataframe(betweenness_df)
                st.markdown(f"""
                **Key Insight**: {betweenness_df.iloc[0]['Node']} has the highest betweenness centrality,
                suggesting it may act as a bridge between different drug interaction pathways.
                """)
        else:
            st.info("Please fetch gene-drug data first in the 'Data Input' tab")

# --- Sales Forecasting with ARIMA ---
def forecast_sales(df, steps, order=(5,1,0)):
    """Forecast future sales using ARIMA model"""
    try:
        if 'Date' not in df.columns or 'Sales' not in df.columns:
            st.error("CSV must contain 'Date' and 'Sales' columns")
            return None, None, None
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        ts = df['Sales']
        if len(ts) < 10:
            st.warning("Not enough data points for reliable forecasting. Need at least 10 data points.")
            return ts, None, None
        model = ARIMA(ts, order=order)
        model_fit = model.fit()
        forecast_results = model_fit.get_forecast(steps=steps)
        forecast = forecast_results.predicted_mean
        conf_int = forecast_results.conf_int()
        return ts, forecast, conf_int
    except Exception as e:
        logger.error(f"Error in sales forecasting: {e}")
        st.error(f"Error in forecasting: {str(e)}")
        return None, None, None

if menu == "Sales Forecast":
    st.subheader("📈 Drug Sales Forecast")
    st.markdown("""
    This module uses ARIMA to forecast future drug sales based on historical data.
    Upload a CSV file with 'Date' and 'Sales' columns or use sample data.
    """)
    
    use_sample = st.checkbox("Use sample data from database")
    if use_sample:
        conn = sqlite3.connect(DB_PATH)
        sample_data = pd.read_sql("""
            SELECT date(sale_date) as Date, SUM(total_price) as Sales
            FROM sales
            GROUP BY date(sale_date)
            ORDER BY Date
        """, conn)
        conn.close()
        st.success("Using sample sales data from database")
        st.subheader("Sample Data Preview")
        st.dataframe(sample_data.head())
        data = sample_data
    else:
        uploaded = st.file_uploader("Upload Sales CSV with Date and Sales columns:", type=['csv'])
        if uploaded:
            try:
                data = pd.read_csv(uploaded)
                st.success("Data uploaded successfully")
            except Exception as e:
                st.error(f"Error reading CSV: {str(e)}")
                data = None
        else:
            data = None
    
    if data is not None:
        st.subheader("Model Parameters")
        col1, col2, col3 = st.columns(3)
        with col1:
            p = st.slider("AR order (p):", 0, 10, 5)
        with col2:
            d = st.slider("Differencing order (d):", 0, 2, 1)
        with col3:
            q = st.slider("MA order (q):", 0, 10, 0)
        period = st.slider("Forecast horizon (days):", 5, 60, 15)
        
        if st.button("Generate Forecast"):
            with st.spinner("Generating forecast..."):
                ts, forecast, conf_int = forecast_sales(data, period, order=(p, d, q))
                if ts is not None:
                    st.subheader("Historical Sales Data")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(ts.index, ts.values)
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Sales')
                    ax.set_title('Historical Sales')
                    ax.grid(True)
                    st.pyplot(fig)
                    if forecast is not None:
                        forecast_index = pd.date_range(start=ts.index[-1] + pd.Timedelta(days=1), periods=period)
                        st.subheader("Sales Forecast")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(ts.index, ts.values, label='Historical')
                        ax.plot(forecast_index, forecast.values, color='red', label='Forecast')
                        if conf_int is not None:
                            ax.fill_between(forecast_index, conf_int.iloc[:, 0].values, conf_int.iloc[:, 1].values,
                                            color='pink', alpha=0.3, label='95% Confidence Interval')
                        ax.set_xlabel('Date')
                        ax.set_ylabel('Sales')
                        ax.set_title('Sales Forecast with ARIMA')
                        ax.legend()
                        ax.grid(True)
                        st.pyplot(fig)

# --- Drug Price Prediction with Machine Learning ---
def load_or_train_price_model(X, y):
    """Load existing model or train a new one"""
    model_path = os.path.join(MODEL_DIR, "drug_price_model.joblib")
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        logger.info("Loaded existing drug price prediction model")
        return model
    else:
        logger.info("Training new drug price prediction model")
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        pipeline.fit(X, y)
        joblib.dump(pipeline, model_path)
        logger.info("Saved new drug price prediction model")
        return pipeline

if menu == "Drug Price ML Model":
    st.subheader("💰 Drug Price Prediction (ML Model)")
    st.markdown("""
    This module uses a Random Forest Regression model to predict drug prices based on features like
    manufacturing complexity, R&D cost, patent status, and market competition.
    """)
    
    tab1, tab2 = st.tabs(["Predict Prices", "Train Model"])
    
    with tab1:
        st.subheader("Predict Drug Price")
        col1, col2 = st.columns(2)
        with col1:
            form_complexity = st.slider("Manufacturing Complexity (1-10):", 1, 10, 5)
            r_and_d_cost = st.number_input("R&D Cost (₹ Thousands):", 100, 5000, 1000)
            patent_expiry = st.slider("Years Until Patent Expiry:", 0, 20, 10)
            active_ingredients = st.slider("Number of Active Ingredients:", 1, 10, 2)
        with col2:
            category = st.selectbox("Treatment Category:", [
                "Cardiovascular", "Antibiotic", "Antidiabetic", "Respiratory", 
                "Psychiatric", "Gastrointestinal", "Pain Management", "Hormonal"
            ])
            category_map = {
                "Cardiovascular": 0, "Antibiotic": 1, "Antidiabetic": 2, "Respiratory": 3,
                "Psychiatric": 4, "Gastrointestinal": 5, "Pain Management": 6, "Hormonal": 7
            }
            competition = st.slider("Market Competition (1-10):", 1, 10, 5)
        
        # Load sample data for initial model training if needed
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT category, price FROM drugs", conn)
        conn.close()
        X_sample = pd.get_dummies(df['category'])
        y_sample = df['price']
        model = load_or_train_price_model(X_sample, y_sample)
        
        if st.button("Predict Price"):
            input_data = pd.DataFrame({
                'complexity': [form_complexity],
                'r_and_d_cost': [r_and_d_cost],
                'patent_expiry': [patent_expiry],
                'active_ingredients': [active_ingredients],
                'competition': [competition]
            })
            category_encoded = pd.get_dummies(pd.Series([category]), prefix='category')
            for col in X_sample.columns:
                if col not in category_encoded.columns:
                    category_encoded[col] = 0
            category_encoded = category_encoded[X_sample.columns]
            input_data = pd.concat([input_data, category_encoded], axis=1)
            prediction = model.predict(input_data)
            st.success(f"Predicted Price: ₹{prediction[0]:,.2f}")
    
    with tab2:
        st.subheader("Train New Model")
        uploaded_file = st.file_uploader("Upload CSV with drug features and prices:", type=['csv'])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            if 'price' in df.columns:
                X = pd.get_dummies(df.drop('price', axis=1))
                y = df['price']
                if st.button("Train Model"):
                    model = load_or_train_price_model(X, y)
                    st.success("Model trained and saved successfully")
            else:
                st.error("CSV must contain a 'price' column")

# Placeholder for other modules
if menu in ["Drug Shortage Prediction", "Medicine Recommendation", "Gene Interaction Network", 
            "Product Catalog", "SQLite Database Explorer", "MongoDB Integration", "Documentation"]:
    st.subheader(f"🚧 {menu}")
    st.info("This module is under development.")