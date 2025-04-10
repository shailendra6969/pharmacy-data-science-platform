"""
Pharmacy Data Science Platform - Main Application

This is the main entry point for the Streamlit web application.
"""
import streamlit as st
import os
import sys
import logging

# Ensure modules in the project directory can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import configuration
from config import MODULES, APP_TITLE, logger

# Import database handlers
from db.sqlite_db import check_and_initialize_db

# Import data loader
from utils.data_loader import load_csv_data

# Import modules
from modules.dashboard import show_dashboard
from modules.gene_analysis import show_gene_analysis
from modules.sales_forecast import show_sales_forecast
from modules.price_prediction import show_price_prediction
from modules.drug_shortage import show_drug_shortage
from modules.recommendation import show_medicine_recommendation
from modules.gene_interaction import show_gene_interaction
from modules.database_explorer import show_database_explorer
from modules.mongo_integration import show_mongo_integration
from modules.documentation import show_documentation

def main():
    """Main application function"""
    # Set page config
    st.set_page_config(layout="wide", page_title=APP_TITLE)
    
    # Title and description
    st.title("💊 Advanced Pharmacy Data Science Platform")

    st.markdown("""
    This professional analytics platform demonstrates data science expertise in pharmaceutical domain with:
    - *Machine Learning*: Drug price prediction models and shortage forecasting
    - *Natural Language Processing*: Medical text analysis and recommendation engine
    - *Network Analysis*: Gene-drug interaction visualization with centrality metrics
    - *Time Series Analysis*: Sales forecasting with ARIMA and seasonal decomposition
    - *Database Integration*: MongoDB and SQLite with real-time querying
    - *Unit Testing*: Comprehensive test suite for data integrity and model validation
    - *Documentation*: Detailed usage guides, API references, and methodologies
    """)
    
    # Load data from CSV
    drugs_df = load_csv_data()
    
    # Initialize database
    check_and_initialize_db(drugs_df)
    
    # Create sidebar menu
    menu = st.sidebar.radio("Select Module", MODULES)
    
    # Display selected module
    if menu == "Dashboard":
        show_dashboard()
    elif menu == "Gene–Drug Analysis":
        show_gene_analysis()
    elif menu == "Sales Forecast":
        show_sales_forecast()
    elif menu == "Drug Price ML Model":
        show_price_prediction()
    elif menu == "Drug Shortage Prediction":
        show_drug_shortage()
    elif menu == "Medicine Recommendation":
        show_medicine_recommendation()
    elif menu == "Gene Interaction Network":
        show_gene_interaction()
    elif menu == "Product Catalog":
        # Placeholder for Product Catalog
        st.subheader("Product Catalog")
        st.info("This module is currently under development.")
    elif menu == "SQLite Database Explorer":
        show_database_explorer()
    elif menu == "MongoDB Integration":
        show_mongo_integration()
    elif menu == "Documentation":
        show_documentation()

if __name__ == "__main__":
    main()
