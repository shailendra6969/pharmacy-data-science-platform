"""
Pharmacy Data Science Platform - Main Application

This is the main entry point for the Streamlit web application.
Includes enhanced features for comprehensive drug data, MongoDB integration,
gene interaction analysis, real-time updates, and medicine verification.
"""
import streamlit as st
import os
import sys
import logging
import atexit
from datetime import datetime

# Ensure modules in the project directory can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import configuration
from config import MODULES, APP_TITLE, logger

# Import database handlers
from db.sqlite_db import check_and_initialize_db
from db.mongo_db import get_mongodb_handler

# Import data loader
from utils.data_loader import load_csv_data
from utils.drug_data_generator import generate_comprehensive_drug_database
from utils.real_time_data import on_app_start, on_app_stop

# Import modules
from modules.dashboard import show_dashboard
from modules.gene_analysis import show_gene_analysis
from modules.sales_forecast import show_sales_forecast
from modules.price_prediction import show_price_prediction
from modules.drug_shortage import show_drug_shortage
from modules.recommendation import show_medicine_recommendation
from modules.gene_interaction import show_gene_interaction
from modules.product_catalog import show_product_catalog
from modules.database_explorer import show_database_explorer
from modules.mongo_integration import show_mongo_integration
from modules.documentation import show_documentation
from modules.medicine_verification import show_medicine_verification

# Register cleanup function
atexit.register(on_app_stop)

def generate_comprehensive_data():
    """Generate comprehensive drug data if it doesn't exist"""
    from config import DATA_DIR
    import os
    
    # Define output path
    output_path = os.path.join(DATA_DIR, "comprehensive_drugs.csv")
    
    # Check if file exists
    if not os.path.exists(output_path):
        st.info("Generating comprehensive drug database (1000+ entries). This may take a moment...")
        progress_bar = st.progress(0)
        
        # Generate the data
        from utils.drug_data_generator import generate_comprehensive_drug_database
        drugs_df = generate_comprehensive_drug_database(1000, output_path)
        
        progress_bar.progress(100)
        st.success(f"Generated comprehensive drug database with {len(drugs_df)} entries!")
        
        return drugs_df
    else:
        # Load existing data
        drugs_df = pd.read_csv(output_path)
        logger.info(f"Loaded existing comprehensive drug database with {len(drugs_df)} entries")
        return drugs_df

def main():
    """Main application function"""
    # Set page config
    st.set_page_config(layout="wide", page_title=APP_TITLE)
    
    # Initialize session state for real-time data
    if 'real_time_initialized' not in st.session_state:
        on_app_start()
        st.session_state.real_time_initialized = True
    
    # Title and description
    st.title("💊 Advanced Pharmacy Data Science Platform")

    st.markdown("""
    This professional analytics platform demonstrates data science expertise in pharmaceutical domain with:
    - *Machine Learning*: Drug price prediction models and shortage forecasting
    - *Natural Language Processing*: Medical text analysis and recommendation engine
    - *Network Analysis*: Gene-drug interaction visualization with centrality metrics
    - *Time Series Analysis*: Sales forecasting with ARIMA and seasonal decomposition
    - *Database Integration*: MongoDB and SQLite with real-time querying
    - *Real-Time Updates*: Live monitoring of prices, stock, and sales
    - *Unit Testing*: Comprehensive test suite for data integrity and model validation
    - *Documentation*: Detailed usage guides, API references, and methodologies
    """)
    
    try:
        # Create action buttons in sidebar
        with st.sidebar:
            st.subheader("Data Management")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Refresh Data"):
                    st.session_state.data_refreshed = True
                    from utils.real_time_data import force_data_update
                    force_data_update("all")
                    st.success("Data refreshed!")
            
            with col2:
                if st.button("Generate Data"):
                    drugs_df = generate_comprehensive_data()
                    # Update database
                    check_and_initialize_db(drugs_df)
                    st.session_state.data_generated = True
        
        # Load data from CSV (enhanced drug data)
        try:
            # Try to load comprehensive data first
            try:
                from config import DATA_DIR
                import pandas as pd
                comprehensive_path = os.path.join(DATA_DIR, "comprehensive_drugs.csv")
                
                if os.path.exists(comprehensive_path):
                    drugs_df = pd.read_csv(comprehensive_path)
                    logger.info(f"Loaded comprehensive drug database with {len(drugs_df)} entries")
                else:
                    # Fall back to regular data loader
                    drugs_df = load_csv_data()
            except Exception as e:
                logger.warning(f"Error loading comprehensive data: {e}")
                # Fall back to regular data loader
                drugs_df = load_csv_data()
            
            # Initialize database
            check_and_initialize_db(drugs_df)
        except Exception as e:
            st.error(f"Error loading initial data: {str(e)}")
            logger.error(f"Error in data initialization: {e}")
            drugs_df = None
    
        # Create sidebar menu with additional modules
        menu = st.sidebar.radio("Select Module", MODULES)
        
        # Display current date/time in sidebar
        st.sidebar.markdown(f"**Current Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Add MongoDB status indicator
        mongo_handler = get_mongodb_handler()
        if mongo_handler.is_connected():
            st.sidebar.success("✅ MongoDB: Connected")
        else:
            st.sidebar.warning("⚠️ MongoDB: Disconnected")
        
        # Add real-time data status
        from utils.real_time_data import get_update_status
        status = get_update_status()
        
        if status["worker_running"]:
            st.sidebar.success("✅ Real-time updates: Active")
        else:
            st.sidebar.warning("⚠️ Real-time updates: Inactive")
        
        # Display selected module
        try:
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
                show_product_catalog()
            elif menu == "SQLite Database Explorer":
                show_database_explorer()
            elif menu == "MongoDB Integration":
                show_mongo_integration()
            elif menu == "Documentation":
                show_documentation()
            elif menu == "Medicine Verification":
                show_medicine_verification()
        except Exception as e:
            st.error(f"Error displaying module '{menu}': {str(e)}")
            logger.error(f"Error in module '{menu}': {e}")
            
            # Show technical details in expander
            with st.expander("Technical Details"):
                st.code(f"Error Type: {type(e).__name__}\nError Message: {str(e)}")
                
                # Suggest solutions based on error type
                if "no such column" in str(e).lower():
                    st.info("""
                    **Possible Solution**: This error usually occurs when the database schema has been updated.
                    Try regenerating the data by clicking the 'Generate Data' button in the sidebar.
                    """)
                elif "connection" in str(e).lower():
                    st.info("""
                    **Possible Solution**: This appears to be a database connection issue.
                    Check that the database file exists and has proper permissions.
                    """)
                elif "import" in str(e).lower():
                    st.info("""
                    **Possible Solution**: This is likely a missing dependency.
                    Make sure you have installed all required packages from requirements.txt.
                    """)
    
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Application error: {e}")

if __name__ == "__main__":
    main()