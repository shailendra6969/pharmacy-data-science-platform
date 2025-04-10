"""
Documentation module for the Pharmacy Data Science Platform.
"""
import streamlit as st
import pandas as pd
from config import APP_VERSION, logger

def show_documentation():
    """Display the platform documentation"""
    st.subheader("📚 Platform Documentation")
    st.markdown(f"""
    ## Pharmacy Data Science Platform v{APP_VERSION}
    
    Welcome to the comprehensive documentation for the Pharmacy Data Science Platform. 
    This guide provides detailed information about each module, their features, and usage instructions.
    """)
    
    # Create tabs for different documentation sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Getting Started", "Module Guide", "API Reference", "Data Dictionary", "FAQ"
    ])
    
    # ----- Getting Started Tab -----
    with tab1:
        st.subheader("Getting Started")
        
        st.markdown("""
        ### Overview
        
        The Pharmacy Data Science Platform is a comprehensive analytics solution designed for pharmaceutical 
        data analysis and visualization. It combines data science techniques, machine learning models, 
        and domain-specific tools to provide insights into drug sales, pricing, shortages, and interactions.
        
        ### Key Features
        
        - **Interactive Dashboard**: Real-time monitoring of key pharmacy metrics
        - **Machine Learning Models**: Price prediction and shortage forecasting
        - **Gene Analysis**: Interaction networks and pathway analysis for pharmacogenomics
        - **Database Integration**: SQLite and MongoDB for structured and unstructured data
        - **Natural Language Processing**: Medicine recommendation engine based on multiple factors
        - **Time Series Analysis**: Sales forecasting with ARIMA models
        
        ### System Requirements
        
        - Python 3.8+
        - Required libraries (all listed in `requirements.txt`)
        - Minimum 4GB RAM recommended for larger datasets
        - SQLite (included by default in Python)
        - MongoDB (optional, for advanced document storage)
        
        ### Installation Guide
        
        1. **Clone the repository**:
           ```bash
           git clone https://github.com/yourusername/pharmacy-data-science.git
           cd pharmacy-data-science
           ```
        
        2. **Create a virtual environment** (recommended):
           ```bash
           python -m venv venv
           source venv/bin/activate  # On Windows: venv\\Scripts\\activate
           ```
        
        3. **Install dependencies**:
           ```bash
           pip install -r requirements.txt
           ```
        
        4. **Run the application**:
           ```bash
           streamlit run app.py
           ```
        
        ### Initial Setup
        
        On first run, the platform will:
        
        1. Create necessary directories (data, models, logs)
        2. Initialize SQLite database
        3. Generate sample data if none exists
        
        To use your own data, place your CSV files in the `data` directory and update the configuration 
        in `config.py` accordingly.
        """)
    
    # ----- Module Guide Tab -----
    with tab2:
        st.subheader("Module Guide")
        
        st.markdown("""
        This section provides an overview of each module in the platform and how to use them effectively.
        """)
        
        # Create expandable sections for each module
        with st.expander("📊 Dashboard"):
            st.markdown("""
            ### Dashboard
            
            The Dashboard module provides a high-level overview of key pharmacy metrics and KPIs.
            
            #### Features:
            - **Key Metrics**: Total products, sales, popular items, and low stock alerts
            - **Sales Trends**: Daily sales with 7-day moving average
            - **Top Products**: Revenue breakdown by product
            - **Category Analysis**: Sales distribution by therapeutic category
            
            #### Usage Tips:
            - Use the time period filter to focus on specific date ranges
            - Hover over charts for detailed information
            - Monitor low stock items to prevent shortages
            """)
        
        with st.expander("🧬 Gene–Drug Analysis"):
            st.markdown("""
            ### Gene–Drug Analysis
            
            This module analyzes interactions between genes and pharmaceutical compounds using data from PharmGKB.
            
            #### Features:
            - **Data Input**: Upload CSV or enter gene IDs directly
            - **API Integration**: Fetch gene-drug interaction data
            - **Visualization**: Interactive gene-drug interaction network
            - **Analysis**: Network centrality metrics to identify key genes
            
            #### Usage Tips:
            - Use valid PharmGKB gene IDs for best results
            - CSV uploads should include a 'Gene ID' column
            - Network visualization can be customized with different layouts
            """)
        
        with st.expander("📈 Sales Forecast"):
            st.markdown("""
            ### Sales Forecast
            
            This module uses time series analysis to predict future drug sales based on historical data.
            
            #### Features:
            - **ARIMA Modeling**: Adjustable parameters for autoregression, differencing, and moving average
            - **Multiple Data Sources**: Database samples or custom CSV uploads
            - **Category Filtering**: Forecast sales for specific drug categories
            - **Visualization**: Historical vs. predicted sales with confidence intervals
            
            #### Usage Tips:
            - Adjust the ARIMA parameters to improve forecast accuracy
            - Higher p values capture long-term trends
            - Higher d values help with non-stationary data
            - Higher q values incorporate short-term fluctuations
            - At least 30 data points are recommended for reliable forecasting
            """)
        
        with st.expander("💰 Drug Price ML Model"):
            st.markdown("""
            ### Drug Price ML Model
            
            This module uses machine learning to predict drug prices based on various factors.
            
            #### Features:
            - **Price Prediction**: Estimate prices based on multiple features
            - **Model Training**: Train custom models with your own data
            - **Feature Importance**: Visualize factors affecting drug pricing
            - **Performance Metrics**: Evaluate model accuracy with key statistics
            
            #### Usage Tips:
            - For accurate predictions, include features like complexity, R&D cost, and competition
            - The model performs best when trained on domain-specific data
            - Review feature importance to understand price drivers
            - Test different parameter combinations to optimize predictions
            """)
        
        with st.expander("🔍 Drug Shortage Prediction"):
            st.markdown("""
            ### Drug Shortage Prediction
            
            This module analyzes inventory and sales data to predict potential drug shortages.
            
            #### Features:
            - **Risk Assessment**: Current shortage risk by product
            - **Predictive Modeling**: Machine learning to forecast future shortages
            - **Seasonal Analysis**: Identify seasonal patterns in drug shortages
            - **Reorder Recommendations**: Suggested quantities for at-risk products
            
            #### Usage Tips:
            - Regularly review the Critical and High Risk items
            - Consider both current inventory and historical sales patterns
            - Pay attention to seasonal trends for better inventory planning
            - Export the reorder list for procurement planning
            """)
        
        with st.expander("💊 Medicine Recommendation"):
            st.markdown("""
            ### Medicine Recommendation
            
            This module uses NLP and content-based filtering to recommend similar medicines and alternatives.
            
            #### Features:
            - **Drug Similarity**: Find similar drugs based on characteristics
            - **Patient Profiles**: Personalized recommendations based on patient data
            - **Sales Patterns**: Co-purchase analysis and bundle suggestions
            - **Price Comparison**: Compare alternatives at different price points
            
            #### Usage Tips:
            - The similarity search works best with comprehensive drug descriptions
            - Include medical conditions for more targeted patient recommendations
            - Review clinical considerations for special populations
            - Bundle offers can be used for marketing promotions
            """)
        
        with st.expander("🧬 Gene Interaction Network"):
            st.markdown("""
            ### Gene Interaction Network
            
            This module visualizes and analyzes gene-gene interaction networks and related pathways.
            
            #### Features:
            - **Network Visualization**: Interactive gene interaction graphs
            - **Centrality Analysis**: Identify key genes in the network
            - **Pathway Analysis**: Discover enriched biological pathways
            - **Filtering Options**: Focus on specific interaction types or confidence levels
            
            #### Usage Tips:
            - Use the visualization options to highlight different aspects of the network
            - Focus on high centrality genes as potential therapeutic targets
            - Review pathway interactions to understand functional relationships
            - Filter by confidence level to focus on well-established interactions
            """)
        
        with st.expander("📋 Product Catalog"):
            st.markdown("""
            ### Product Catalog
            
            This module provides a comprehensive view of the pharmaceutical product inventory.
            
            #### Features:
            - **Search and Filter**: Find products by name, category, or manufacturer
            - **Detailed Product Views**: Complete information for each product
            - **Analytics**: Price distribution and stock analysis
            - **Export Options**: Download filtered catalog as CSV
            
            #### Usage Tips:
            - Use multiple filters to narrow down product searches
            - Review the analytics section for pricing strategy insights
            - Check product details for sales history and margin analysis
            - Export filtered lists for inventory reports
            """)
        
        with st.expander("🗄️ SQLite Database Explorer"):
            st.markdown("""
            ### SQLite Database Explorer
            
            This module allows direct exploration and querying of the SQLite database.
            
            #### Features:
            - **Schema Viewer**: Database table structure and relationships
            - **SQL Query Tool**: Execute custom SQL queries
            - **Data Visualization**: Chart and graph query results
            - **Export Options**: Download query results as CSV
            
            #### Usage Tips:
            - Review the schema before writing queries
            - Use the sample queries as starting points
            - Only SELECT queries are allowed for safety
            - Visualize results to identify patterns in the data
            """)
        
        with st.expander("📊 MongoDB Integration"):
            st.markdown("""
            ### MongoDB Integration
            
            This module provides integration with MongoDB for unstructured data storage and analysis.
            
            #### Features:
            - **Data Import/Export**: Move data between formats and systems
            - **Document Explorer**: Query and browse MongoDB collections
            - **Text Analysis**: Word frequency and pattern analysis
            - **Time Series Analysis**: Trend visualization for temporal data
            
            #### Usage Tips:
            - Use simulation mode if MongoDB is not available
            - JSON is the preferred format for data exchange
            - Text analysis works best with substantial text content
            - Time series analysis requires timestamp fields
            """)
    
    # ----- API Reference Tab -----
    with tab3:
        st.subheader("API Reference")
        
        st.markdown("""
        This section provides documentation for the internal APIs and functions used by the platform.
        These can be used for extending the platform or integrating with other systems.
        """)
        
        # Database API
        st.markdown("""
        ### Database API
        
        #### SQLite Functions
        
        ```python
        # Get a database connection
        conn = get_db_connection()
        
        # Execute a query with parameters
        results = execute_query("SELECT * FROM table WHERE column = ?", params=("value",))
        
        # Execute a non-SELECT query
        execute_query("UPDATE table SET column = ? WHERE id = ?", 
                     params=("new_value", 1), fetch=False)
        ```
        
        #### MongoDB Functions
        
        ```python
        # Get a MongoDB connection
        client = get_mongo_connection()
        db = client[MONGO_DB]
        
        # Insert a document
        db.collection.insert_one({"key": "value"})
        
        # Query documents
        documents = list(db.collection.find({"key": "value"}))
        
        # Aggregation
        result = db.collection.aggregate([
            {"$group": {"_id": "$category", "count": {"$sum": 1}}}
        ])
        ```
        """)
        
        # Data Loading API
        st.markdown("""
        ### Data Loading API
        
        ```python
        # Load CSV data from the configured path
        df = load_csv_data()
        
        # Load data from an uploaded file
        df = load_uploaded_data(uploaded_file, required_columns=["col1", "col2"])
        ```
        """)
        
        # Machine Learning API
        st.markdown("""
        ### Machine Learning API
        
        #### Price Prediction
        
        ```python
        # Load or train price prediction model
        model = load_or_train_price_model(X, y)
        
        # Evaluate model performance
        metrics, predictions = evaluate_model(model, X_test, y_test)
        ```
        
        #### Sales Forecasting
        
        ```python
        # Forecast future sales with ARIMA
        ts, forecast, conf_int, model = forecast_sales(df, steps=30, order=(5, 1, 0))
        ```
        
        #### Drug Shortage Prediction
        
        ```python
        # Calculate days of inventory
        days = calculate_days_of_inventory(stock, avg_daily_sales)
        
        # Classify shortage risk
        risk = classify_shortage_risk(days_of_inventory)
        ```
        """)
        
        # Network Analysis API
        st.markdown("""
        ### Network Analysis API
        
        ```python
        # Create gene interaction network
        G = create_network_from_dataframe(df, source_col="source", target_col="target")
        
        # Calculate network centrality metrics
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        closeness_centrality = nx.closeness_centrality(G)
        ```
        """)
        
        # Recommendation API
        st.markdown("""
        ### Recommendation API
        
        ```python
        # Get similar drugs based on content
        similar_drugs = get_recommendations(drug_id, content_matrix, indices, drugs_df, n=5)
        
        # Create combined features for similarity calculation
        features = create_combined_features(row)
        ```
        """)
    
    # ----- Data Dictionary Tab -----
    with tab4:
        st.subheader("Data Dictionary")
        
        st.markdown("""
        This section provides details about the data structure, tables, and fields used in the platform.
        """)
        
        # Create table definitions
        drugs_table = {
            "Table": "drugs",
            "Description": "Main table for pharmaceutical products information",
            "Fields": [
                {"Name": "id", "Type": "INTEGER", "Description": "Unique identifier for the drug"},
                {"Name": "name", "Type": "TEXT", "Description": "Name of the drug"},
                {"Name": "category", "Type": "TEXT", "Description": "Therapeutic category (e.g., Antibiotic, Cardiovascular)"},
                {"Name": "price", "Type": "REAL", "Description": "Current selling price"},
                {"Name": "dosage", "Type": "TEXT", "Description": "Recommended dosage information"},
                {"Name": "description", "Type": "TEXT", "Description": "Detailed description of the drug"},
                {"Name": "manufacturer", "Type": "TEXT", "Description": "Drug manufacturer name"},
                {"Name": "stock", "Type": "INTEGER", "Description": "Current inventory level"}
            ]
        }
        
        sales_table = {
            "Table": "sales",
            "Description": "Transaction records for drug sales",
            "Fields": [
                {"Name": "id", "Type": "INTEGER", "Description": "Unique identifier for the sale"},
                {"Name": "drug_id", "Type": "INTEGER", "Description": "Foreign key reference to drugs.id"},
                {"Name": "sale_date", "Type": "TEXT", "Description": "Date of the sale (YYYY-MM-DD format)"},
                {"Name": "quantity", "Type": "INTEGER", "Description": "Number of units sold"},
                {"Name": "total_price", "Type": "REAL", "Description": "Total revenue from the sale"}
            ]
        }
        
        # Display table definitions
        with st.expander("SQLite Database Schema"):
            # Drugs table
            st.markdown(f"### {drugs_table['Table']}")
            st.markdown(drugs_table['Description'])
            
            drugs_df = pd.DataFrame(drugs_table['Fields'])
            st.table(drugs_df)
            
            # Sales table
            st.markdown(f"### {sales_table['Table']}")
            st.markdown(sales_table['Description'])
            
            sales_df = pd.DataFrame(sales_table['Fields'])
            st.table(sales_df)
            
            # Relationships
            st.markdown("""
            ### Relationships
            
            - **sales.drug_id** → **drugs.id** (Many-to-One)
            """)
        
        with st.expander("MongoDB Collections"):
            st.markdown("""
            ### patient_feedback
            
            Collection for storing patient feedback and ratings for medications.
            
            #### Sample Document:
            ```json
            {
                "patient_id": "P1001",
                "drug_name": "Lisinopril",
                "rating": 4,
                "effectiveness": 5,
                "side_effects": 2,
                "comments": "Very effective medication. Helped with my symptoms quickly.",
                "timestamp": "2023-10-15T14:30:00",
                "verified_purchase": true
            }
            ```
            
            ### clinical_notes
            
            Collection for storing physician notes and observations about drug prescriptions.
            
            #### Sample Document:
            ```json
            {
                "patient_id": "P1001",
                "physician_id": "DR105",
                "drug_prescribed": "Lisinopril",
                "condition": "Hypertension",
                "dosage": "10 mg",
                "frequency": "once daily",
                "notes": "Patient presenting with Hypertension. Prescribed Lisinopril for treatment. Patient responding well to treatment.",
                "visit_date": "2023-09-20T10:15:00",
                "follow_up_recommended": true
            }
            ```
            
            ### drug_interactions
            
            Collection for storing information about interactions between different medications.
            
            #### Sample Document:
            ```json
            {
                "primary_drug": "Warfarin",
                "secondary_drug": "Aspirin",
                "severity": "Major",
                "mechanism": "Additive Effects",
                "recommendation": "Avoid combination",
                "evidence_level": "Strong",
                "references": ["Reference 42", "Reference 87"],
                "last_updated": "2023-05-12T08:45:00"
            }
            ```
            
            ### adverse_events
            
            Collection for storing reports of adverse drug reactions.
            
            #### Sample Document:
            ```json
            {
                "report_id": "AE10005",
                "patient_age": 65,
                "patient_gender": "Female",
                "drug_name": "Atorvastatin",
                "event_description": "Muscle pain and stiffness",
                "onset_date": "2023-10-01T00:00:00",
                "report_date": "2023-10-05T14:30:00",
                "severity": "Moderate",
                "outcome": "Recovering",
                "causality_assessment": "Probable",
                "concomitant_medications": true,
                "reporter_type": "Physician"
            }
            ```
            """)
        
        with st.expander("CSV Data Format"):
            st.markdown("""
            ### sample_100_drugs.csv
            
            This file contains information about 100 sample drugs used to populate the database.
            
            #### Columns:
            - **id**: Unique identifier (1-100)
            - **name**: Drug name (e.g., "Drug-001")
            - **category**: Therapeutic category
            - **price**: Price in ₹ (Indian Rupees)
            - **dosage**: Recommended dosage
            - **description**: Product description
            - **manufacturer**: Company name
            - **stock**: Current inventory level
            
            ### gene_data.csv (Optional Upload)
            
            Format for uploading gene interaction data.
            
            #### Required Columns:
            - **Gene ID**: PharmGKB gene identifier (e.g., "PA124")
            
            ### sales_data.csv (Optional Upload)
            
            Format for uploading sales forecasting data.
            
            #### Required Columns:
            - **Date**: Date in YYYY-MM-DD format
            - **Sales**: Numeric sales value
            """)
        
        with st.expander("Model Data Formats"):
            st.markdown("""
            ### Drug Price Model
            
            The price prediction model is stored as a joblib file with the following components:
            
            - **Preprocessing**: StandardScaler for feature normalization
            - **Model**: RandomForestRegressor with configurable parameters
            - **Features**: Typically includes categorical features (one-hot encoded) and numerical features
            
            ### Drug Shortage Model
            
            The shortage prediction model is stored as a joblib file with these components:
            
            - **Preprocessing**: StandardScaler for feature normalization
            - **Model**: Classification model (RandomForest or LogisticRegression)
            - **Features**: Inventory metrics, sales volatility, and historical patterns
            """)
    
    # ----- FAQ Tab -----
    with tab5:
        st.subheader("Frequently Asked Questions")
        
        # General FAQs
        with st.expander("General Questions"):
            st.markdown("""
            #### Q: What is the Pharmacy Data Science Platform?
            A: It's a comprehensive analytics solution specifically designed for pharmaceutical data analysis, combining database management, machine learning, visualization, and domain-specific tools in one integrated platform.
            
            #### Q: Do I need programming knowledge to use the platform?
            A: No, the platform is designed with a user-friendly interface that doesn't require programming knowledge. However, familiarity with SQL can be helpful for the Database Explorer module.
            
            #### Q: Can I use my own data with the platform?
            A: Yes, the platform supports importing your own data through CSV uploads. You can also configure the paths in config.py to point to your data files.
            
            #### Q: Is my data secure?
            A: Yes, the platform runs locally on your machine, so your data never leaves your system unless you configure external database connections.
            """)
        
        # Technical FAQs
        with st.expander("Technical Questions"):
            st.markdown("""
            #### Q: What should I do if I encounter an error?
            A: Check the logs in the 'logs' directory. Most errors are recorded there with detailed information that can help diagnose the issue.
            
            #### Q: How do I update the platform?
            A: Pull the latest changes from the repository and restart the application. Your data will be preserved unless there are major schema changes.
            
            #### Q: Can I run the platform on a server for multi-user access?
            A: Yes, you can deploy Streamlit applications on a server. Refer to the Streamlit documentation for deployment options.
            
            #### Q: How do I back up my data?
            A: The SQLite database is stored in the 'data' directory. You can simply copy this file to create a backup. For MongoDB data, use the export functionality in the MongoDB Integration module.
            """)
        
        # Module-specific FAQs
        with st.expander("Module-Specific Questions"):
            st.markdown("""
            #### Q: Why is my sales forecast not accurate?
            A: ARIMA model accuracy depends on sufficient historical data and appropriate parameter selection. Try adjusting the p, d, and q parameters and ensure you have at least 30 data points.
            
            #### Q: How does the drug similarity recommendation work?
            A: It uses TF-IDF vectorization and cosine similarity to compare drug characteristics like category, description, and manufacturer, finding items with similar profiles.
            
            #### Q: Why can't I connect to MongoDB?
            A: Check that MongoDB is installed and running on your system. The connection URI in config.py should point to your MongoDB instance. You can use Simulation Mode if MongoDB is not available.
            
            #### Q: How often should I check the Drug Shortage Prediction module?
            A: It's recommended to review it weekly, paying special attention to items in the Critical and High Risk categories.
            """)
        
        # Troubleshooting
        with st.expander("Troubleshooting"):
            st.markdown("""
            #### Connection Issues
            - **SQLite**: Ensure the database path in config.py is correct and accessible
            - **MongoDB**: Verify MongoDB is running and the connection URI is correct
            
            #### Import Errors
            - Ensure all dependencies listed in requirements.txt are installed
            - Check for version conflicts between packages
            
            #### Performance Issues
            - Large datasets may cause slowdowns; consider filtering or sampling data
            - For MongoDB queries, ensure proper indexing for better performance
            
            #### Visualization Problems
            - If charts don't display correctly, try adjusting the browser window size
            - For complex visualizations, increase the figure size parameters
            """)