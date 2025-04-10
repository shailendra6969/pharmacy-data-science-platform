"""
MongoDB Integration module for the Pharmacy Data Science Platform.
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import datetime
from pymongo import MongoClient
from bson.json_util import dumps, loads
from config import logger, MONGO_URI, MONGO_DB, MONGO_CONN_TIMEOUT
from db.sqlite_db import execute_query

def get_mongo_connection():
    """Establish connection to MongoDB with error handling"""
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=MONGO_CONN_TIMEOUT)
        # Verify connection works
        client.server_info()
        return client
    except Exception as e:
        logger.error(f"MongoDB connection error: {e}")
        return None

def show_mongo_integration():
    """Display the MongoDB Integration module"""
    st.subheader("📊 MongoDB Integration")
    st.markdown("""
    This module provides integration with MongoDB for storing and analyzing unstructured data,
    such as patient feedback, clinical notes, and drug interaction reports.
    """)
    
    # Check MongoDB connection
    client = get_mongo_connection()
    
    if client is None:
        st.error("""
        Could not connect to MongoDB server. Please check that:
        1. MongoDB is installed and running
        2. Connection URI in config.py is correct
        3. Network allows connection to the MongoDB server
        """)
        
        # Offer simulation mode
        st.info("""
        Would you like to run in simulation mode?
        This will demonstrate MongoDB functionality using mock data.
        """)
        
        simulation_mode = st.checkbox("Use simulation mode", value=True)
        
        if not simulation_mode:
            return
    else:
        simulation_mode = False
        db = client[MONGO_DB]
    
    # Create tabs for different MongoDB functions
    tab1, tab2, tab3 = st.tabs(["Data Import/Export", "Document Explorer", "Analytics"])
    
    # ----- Data Import/Export Tab -----
    with tab1:
        st.subheader("Data Import & Export")
        
        if not simulation_mode:
            # Get collection names
            collections = db.list_collection_names()
            
            if not collections:
                st.info("No collections found in the database. Create a new collection to get started.")
            else:
                st.success(f"Found {len(collections)} collections in the database.")
        else:
            # Simulated collections for demonstration
            collections = ["patient_feedback", "clinical_notes", "drug_interactions", "adverse_events"]
            st.info("SIMULATION MODE: Using mock collections for demonstration.")
        
        # Collection selection or creation
        collection_action = st.radio("Select Action:", ["Use Existing Collection", "Create New Collection"])
        
        if collection_action == "Use Existing Collection":
            if collections:
                selected_collection = st.selectbox("Select Collection:", collections)
            else:
                st.warning("No existing collections found. Please create a new collection.")
                selected_collection = None
        else:
            new_collection_name = st.text_input("New Collection Name:")
            
            if st.button("Create Collection") and new_collection_name:
                if not simulation_mode:
                    try:
                        # Create new collection
                        db.create_collection(new_collection_name)
                        st.success(f"Collection '{new_collection_name}' created successfully.")
                        selected_collection = new_collection_name
                        collections = db.list_collection_names()  # Refresh list
                    except Exception as e:
                        logger.error(f"Error creating collection: {e}")
                        st.error(f"Error creating collection: {str(e)}")
                        selected_collection = None
                else:
                    st.success(f"SIMULATION: Collection '{new_collection_name}' created.")
                    collections.append(new_collection_name)
                    selected_collection = new_collection_name
            else:
                selected_collection = None
        
        # Data Import Options
        if selected_collection:
            st.subheader(f"Import Data to '{selected_collection}'")
            
            import_option = st.radio(
                "Import Source:",
                ["Upload JSON File", "Import from SQLite", "Enter JSON Document"]
            )
            
            if import_option == "Upload JSON File":
                uploaded_file = st.file_uploader("Upload JSON file:", type=["json"])
                
                if uploaded_file is not None:
                    try:
                        # Load JSON data
                        content = uploaded_file.getvalue().decode("utf-8")
                        json_data = json.loads(content)
                        
                        # Display JSON preview
                        st.subheader("Data Preview")
                        if isinstance(json_data, list):
                            st.json(json_data[0] if json_data else {})
                            doc_count = len(json_data)
                            st.info(f"JSON file contains {doc_count} documents.")
                        else:
                            st.json(json_data)
                            doc_count = 1
                            st.info("JSON file contains 1 document.")
                        
                        # Import button
                        if st.button("Import Data"):
                            if not simulation_mode:
                                try:
                                    if isinstance(json_data, list):
                                        result = db[selected_collection].insert_many(json_data)
                                        st.success(f"Successfully imported {len(result.inserted_ids)} documents.")
                                    else:
                                        result = db[selected_collection].insert_one(json_data)
                                        st.success(f"Successfully imported document with ID: {result.inserted_id}")
                                except Exception as e:
                                    logger.error(f"Error importing JSON: {e}")
                                    st.error(f"Error importing data: {str(e)}")
                            else:
                                st.success(f"SIMULATION: Successfully imported {doc_count} documents.")
                    
                    except json.JSONDecodeError:
                        st.error("Invalid JSON format. Please check your file.")
                    
                    except Exception as e:
                        logger.error(f"Error processing JSON file: {e}")
                        st.error(f"Error processing file: {str(e)}")
            
            elif import_option == "Import from SQLite":
                # Get tables from SQLite
                try:
                    sqlite_tables = execute_query("""
                        SELECT name FROM sqlite_master 
                        WHERE type='table' 
                        ORDER BY name
                    """)
                    
                    if sqlite_tables.empty:
                        st.warning("No tables found in SQLite database.")
                    else:
                        table_list = sqlite_tables['name'].tolist()
                        selected_table = st.selectbox("Select SQLite Table:", table_list)
                        
                        # Query to preview data
                        limit = st.slider("Number of records to import:", 1, 1000, 100)
                        
                        # Preview data
                        preview_data = execute_query(f"SELECT * FROM {selected_table} LIMIT 5")
                        if not preview_data.empty:
                            st.subheader("Data Preview")
                            st.dataframe(preview_data)
                            
                            # Import button
                            if st.button("Import Data"):
                                try:
                                    # Get data from SQLite
                                    sqlite_data = execute_query(f"SELECT * FROM {selected_table} LIMIT {limit}")
                                    
                                    if not sqlite_data.empty:
                                        # Convert to list of dictionaries for MongoDB
                                        records = sqlite_data.to_dict(orient='records')
                                        
                                        if not simulation_mode:
                                            try:
                                                # Insert into MongoDB
                                                result = db[selected_collection].insert_many(records)
                                                st.success(f"Successfully imported {len(result.inserted_ids)} records.")
                                            except Exception as e:
                                                logger.error(f"Error importing to MongoDB: {e}")
                                                st.error(f"Error importing data: {str(e)}")
                                        else:
                                            st.success(f"SIMULATION: Successfully imported {len(records)} records.")
                                    else:
                                        st.warning("No data to import.")
                                
                                except Exception as e:
                                    logger.error(f"Error processing SQLite data: {e}")
                                    st.error(f"Error: {str(e)}")
                        else:
                            st.warning("No data found in the selected table.")
                
                except Exception as e:
                    logger.error(f"Error accessing SQLite database: {e}")
                    st.error(f"Error accessing SQLite: {str(e)}")
            
            elif import_option == "Enter JSON Document":
                json_text = st.text_area(
                    "Enter JSON Document:",
                    value='{\n  "name": "Sample Drug",\n  "category": "Antibiotic",\n  "effects": ["infection", "pain"],\n  "contraindications": {\n    "pregnancy": true,\n    "liver_disease": false\n  }\n}',
                    height=250
                )
                
                if json_text:
                    try:
                        # Parse JSON
                        json_data = json.loads(json_text)
                        
                        # Add timestamp if not present
                        if 'timestamp' not in json_data:
                            json_data['timestamp'] = datetime.datetime.now().isoformat()
                        
                        # Import button
                        if st.button("Import Document"):
                            if not simulation_mode:
                                try:
                                    result = db[selected_collection].insert_one(json_data)
                                    st.success(f"Successfully imported document with ID: {result.inserted_id}")
                                except Exception as e:
                                    logger.error(f"Error importing document: {e}")
                                    st.error(f"Error importing document: {str(e)}")
                            else:
                                st.success("SIMULATION: Successfully imported document.")
                    
                    except json.JSONDecodeError as e:
                        st.error(f"Invalid JSON format: {str(e)}")
                    
                    except Exception as e:
                        logger.error(f"Error processing JSON input: {e}")
                        st.error(f"Error: {str(e)}")
            
            # Export Data
            st.subheader(f"Export Data from '{selected_collection}'")
            
            if st.button("Export Collection to JSON"):
                if not simulation_mode:
                    try:
                        # Get data from MongoDB
                        cursor = db[selected_collection].find({})
                        documents = list(cursor)
                        
                        if documents:
                            # Convert to JSON
                            json_data = dumps(documents, indent=2)
                            
                            # Provide download button
                            st.download_button(
                                label="Download JSON",
                                data=json_data,
                                file_name=f"{selected_collection}.json",
                                mime="application/json"
                            )
                        else:
                            st.info("No documents found in the collection.")
                    
                    except Exception as e:
                        logger.error(f"Error exporting data: {e}")
                        st.error(f"Error exporting data: {str(e)}")
                else:
                    # Create mock data for simulation
                    if selected_collection == "patient_feedback":
                        mock_data = generate_mock_patient_feedback(10)
                    elif selected_collection == "clinical_notes":
                        mock_data = generate_mock_clinical_notes(10)
                    elif selected_collection == "drug_interactions":
                        mock_data = generate_mock_drug_interactions(10)
                    elif selected_collection == "adverse_events":
                        mock_data = generate_mock_adverse_events(10)
                    else:
                        mock_data = [{"sample": "data", "collection": selected_collection}]
                    
                    # Convert to JSON
                    json_data = json.dumps(mock_data, indent=2)
                    
                    # Provide download button
                    st.download_button(
                        label="Download JSON (Simulated)",
                        data=json_data,
                        file_name=f"{selected_collection}.json",
                        mime="application/json"
                    )
    
    # ----- Document Explorer Tab -----
    with tab2:
        st.subheader("MongoDB Document Explorer")
        
        if not collections:
            st.info("No collections available. Please create or import data first.")
        else:
            # Select collection to explore
            selected_collection = st.selectbox("Select Collection to Explore:", collections, key="explorer_collection")
            
            # Query options
            st.subheader("Query Options")
            
            query_type = st.radio("Query Type:", ["Find All", "Query by Field", "Advanced Query"])
            
            query_limit = st.slider("Result Limit:", 1, 100, 10)
            
            if query_type == "Find All":
                query = {}
                query_display = "{}"
            
            elif query_type == "Query by Field":
                field_name = st.text_input("Field Name:", "name")
                field_value = st.text_input("Field Value:", "Sample Drug")
                
                query = {field_name: field_value}
                query_display = json.dumps(query)
            
            elif query_type == "Advanced Query":
                query_json = st.text_area(
                    "Enter Query JSON:",
                    value='{"category": "Antibiotic", "effects": {"$in": ["infection"]}}',
                    height=100
                )
                
                try:
                    query = json.loads(query_json)
                    query_display = query_json
                except json.JSONDecodeError:
                    st.error("Invalid JSON query format.")
                    query = {}
                    query_display = "{}"
            
            # Execute query
            if st.button("Execute Query"):
                if not simulation_mode:
                    try:
                        # Run MongoDB query
                        cursor = db[selected_collection].find(query).limit(query_limit)
                        results = list(cursor)
                        
                        # Display results
                        if results:
                            st.subheader(f"Query Results ({len(results)} documents)")
                            st.code(f"Query: {query_display}")
                            
                            # Format and display each document
                            for i, doc in enumerate(results):
                                with st.expander(f"Document {i+1}"):
                                    st.json(json.loads(dumps(doc)))
                            
                            # Convert to DataFrame if possible
                            try:
                                df = pd.json_normalize(results)
                                
                                if not df.empty:
                                    st.subheader("Results as Table")
                                    st.dataframe(df)
                                    
                                    # Option to download as CSV
                                    csv = df.to_csv(index=False)
                                    st.download_button(
                                        label="Download CSV",
                                        data=csv,
                                        file_name=f"{selected_collection}_query.csv",
                                        mime="text/csv"
                                    )
                            except Exception as e:
                                logger.error(f"Error converting to DataFrame: {e}")
                                st.warning("Could not convert documents to table format.")
                        else:
                            st.info("No documents match the query.")
                    
                    except Exception as e:
                        logger.error(f"Error executing MongoDB query: {e}")
                        st.error(f"Error executing query: {str(e)}")
                else:
                    # Generate mock results for simulation
                    if selected_collection == "patient_feedback":
                        results = generate_mock_patient_feedback(query_limit)
                    elif selected_collection == "clinical_notes":
                        results = generate_mock_clinical_notes(query_limit)
                    elif selected_collection == "drug_interactions":
                        results = generate_mock_drug_interactions(query_limit)
                    elif selected_collection == "adverse_events":
                        results = generate_mock_adverse_events(query_limit)
                    else:
                        results = [{"sample": "data", "collection": selected_collection}] * query_limit
                    
                    # Display simulation results
                    st.subheader(f"SIMULATION: Query Results ({len(results)} documents)")
                    st.code(f"Query: {query_display}")
                    
                    # Format and display each document
                    for i, doc in enumerate(results):
                        with st.expander(f"Document {i+1}"):
                            st.json(doc)
                    
                    # Convert to DataFrame
                    df = pd.json_normalize(results)
                    
                    st.subheader("Results as Table")
                    st.dataframe(df)
                    
                    # Option to download as CSV
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV (Simulated)",
                        data=csv,
                        file_name=f"{selected_collection}_query.csv",
                        mime="text/csv"
                    )
    
    # ----- Analytics Tab -----
    with tab3:
        st.subheader("MongoDB Analytics")
        
        if not collections:
            st.info("No collections available. Please create or import data first.")
        else:
            # Select collection for analytics
            selected_collection = st.selectbox("Select Collection:", collections, key="analytics_collection")
            
            # Analytics options
            analysis_type = st.radio(
                "Analysis Type:",
                ["Document Count", "Field Distribution", "Text Analysis", "Time Series"]
            )
            
            if analysis_type == "Document Count":
                if st.button("Count Documents"):
                    if not simulation_mode:
                        try:
                            count = db[selected_collection].count_documents({})
                            st.metric("Document Count", count)
                        except Exception as e:
                            logger.error(f"Error counting documents: {e}")
                            st.error(f"Error: {str(e)}")
                    else:
                        # Simulated count
                        st.metric("Document Count (Simulated)", np.random.randint(50, 500))
            
            elif analysis_type == "Field Distribution":
                field_name = st.text_input("Field Name for Distribution Analysis:", "category")
                
                if st.button("Analyze Distribution"):
                    if not simulation_mode:
                        try:
                            # Use MongoDB aggregation
                            pipeline = [
                                {"$group": {"_id": f"${field_name}", "count": {"$sum": 1}}},
                                {"$sort": {"count": -1}}
                            ]
                            
                            results = list(db[selected_collection].aggregate(pipeline))
                            
                            if results:
                                # Convert to DataFrame
                                dist_df = pd.DataFrame(results)
                                dist_df.columns = ['Value', 'Count']
                                
                                # Display as table
                                st.dataframe(dist_df)
                                
                                # Create chart
                                fig, ax = plt.subplots(figsize=(10, 6))
                                dist_df.plot(kind='bar', x='Value', y='Count', ax=ax)
                                plt.title(f"Distribution of '{field_name}'")
                                plt.xlabel(field_name)
                                plt.ylabel("Count")
                                plt.tight_layout()
                                st.pyplot(fig)
                            else:
                                st.info(f"No values found for field '{field_name}'.")
                        
                        except Exception as e:
                            logger.error(f"Error analyzing distribution: {e}")
                            st.error(f"Error: {str(e)}")
                    else:
                        # Simulated distribution
                        if field_name == "category":
                            categories = ["Antibiotic", "Cardiovascular", "Pain Management", 
                                        "Psychiatric", "Respiratory", "Gastrointestinal"]
                            counts = np.random.randint(10, 100, size=len(categories))
                        elif field_name == "rating":
                            categories = [1, 2, 3, 4, 5]
                            counts = [5, 15, 30, 45, 25]  # Distribution skewed toward higher ratings
                        elif field_name == "severity":
                            categories = ["Mild", "Moderate", "Severe"]
                            counts = [50, 30, 10]
                        else:
                            categories = [f"Value {i}" for i in range(1, 6)]
                            counts = np.random.randint(5, 50, size=len(categories))
                        
                        # Create DataFrame
                        dist_df = pd.DataFrame({
                            'Value': categories,
                            'Count': counts
                        })
                        
                        # Display as table
                        st.dataframe(dist_df)
                        
                        # Create chart
                        fig, ax = plt.subplots(figsize=(10, 6))
                        dist_df.plot(kind='bar', x='Value', y='Count', ax=ax)
                        plt.title(f"Distribution of '{field_name}' (Simulated)")
                        plt.xlabel(field_name)
                        plt.ylabel("Count")
                        plt.tight_layout()
                        st.pyplot(fig)
            
            elif analysis_type == "Text Analysis":
                text_field = st.text_input("Text Field to Analyze:", "comments")
                
                if st.button("Analyze Text"):
                    if not simulation_mode:
                        try:
                            # Get all text values from the field
                            pipeline = [
                                {"$match": {text_field: {"$exists": True, "$ne": ""}}},
                                {"$project": {text_field: 1}}
                            ]
                            
                            results = list(db[selected_collection].aggregate(pipeline))
                            
                            if results:
                                # Extract text from results
                                texts = [doc.get(text_field, "") for doc in results]
                                texts = [t for t in texts if t]  # Remove empty strings
                                
                                if texts:
                                    # Simple word frequency analysis
                                    st.subheader(f"Text Analysis of '{text_field}'")
                                    
                                    # Combine all text
                                    all_text = " ".join(texts)
                                    
                                    # Split into words and count
                                    words = all_text.lower().split()
                                    word_counts = pd.Series(words).value_counts().head(20)
                                    
                                    # Display results
                                    st.subheader("Top 20 Words")
                                    
                                    # Create bar chart
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    word_counts.plot(kind='bar', ax=ax)
                                    plt.title(f"Most Common Words in '{text_field}'")
                                    plt.xlabel("Word")
                                    plt.ylabel("Frequency")
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                    
                                    # Show sample texts
                                    st.subheader("Sample Texts")
                                    for i, text in enumerate(texts[:5]):
                                        st.write(f"{i+1}. {text}")
                                else:
                                    st.info(f"No text content found in field '{text_field}'.")
                            else:
                                st.info(f"No documents found with field '{text_field}'.")
                        
                        except Exception as e:
                            logger.error(f"Error analyzing text: {e}")
                            st.error(f"Error: {str(e)}")
                    else:
                        # Simulated text analysis
                        st.subheader(f"Text Analysis of '{text_field}' (Simulated)")
                        
                        # Create simulated word frequencies
                        common_words = ["medication", "effective", "treatment", "symptoms", "doctor", 
                                      "pain", "relief", "pharmacy", "prescription", "dosage",
                                      "effect", "side", "improvement", "better", "worse",
                                      "days", "weeks", "recommend", "patient", "drug"]
                        
                        word_counts = pd.Series({word: np.random.randint(5, 50) for word in common_words})
                        word_counts = word_counts.sort_values(ascending=False)
                        
                        # Create bar chart
                        fig, ax = plt.subplots(figsize=(10, 6))
                        word_counts.plot(kind='bar', ax=ax)
                        plt.title(f"Most Common Words in '{text_field}' (Simulated)")
                        plt.xlabel("Word")
                        plt.ylabel("Frequency")
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Show sample texts
                        st.subheader("Sample Texts (Simulated)")
                        sample_texts = [
                            "The medication was very effective for my symptoms.",
                            "I experienced some side effects, but the overall relief was worth it.",
                            "After two weeks of treatment, I saw significant improvement.",
                            "The dosage prescribed by my doctor seemed too strong initially.",
                            "I would recommend this drug to others with similar conditions."
                        ]
                        
                        for i, text in enumerate(sample_texts):
                            st.write(f"{i+1}. {text}")
            
            elif analysis_type == "Time Series":
                time_field = st.text_input("Timestamp Field:", "timestamp")
                value_field = st.text_input("Value Field:", "rating")
                
                if st.button("Generate Time Series"):
                    if not simulation_mode:
                        try:
                            # Use MongoDB aggregation for time series
                            pipeline = [
                                {"$match": {time_field: {"$exists": True}, value_field: {"$exists": True}}},
                                {"$project": {
                                    "date": {"$dateFromString": {"dateString": f"${time_field}"}},
                                    "value": f"${value_field}"
                                }},
                                {"$group": {
                                    "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$date"}},
                                    "avg_value": {"$avg": "$value"},
                                    "count": {"$sum": 1}
                                }},
                                {"$sort": {"_id": 1}}
                            ]
                            
                            results = list(db[selected_collection].aggregate(pipeline))
                            
                            if results:
                                # Convert to DataFrame
                                ts_df = pd.DataFrame(results)
                                ts_df.columns = ['Date', 'Average Value', 'Count']
                                ts_df['Date'] = pd.to_datetime(ts_df['Date'])
                                ts_df = ts_df.set_index('Date')
                                
                                # Display as table
                                st.dataframe(ts_df.reset_index())
                                
                                # Create chart
                                fig, ax1 = plt.subplots(figsize=(12, 6))
                                
                                # Plot average value
                                ax1.plot(ts_df.index, ts_df['Average Value'], 'b-', marker='o')
                                ax1.set_xlabel('Date')
                                ax1.set_ylabel('Average Value', color='b')
                                ax1.tick_params(axis='y', labelcolor='b')
                                
                                # Create second y-axis for count
                                ax2 = ax1.twinx()
                                ax2.plot(ts_df.index, ts_df['Count'], 'r--', marker='x')
                                ax2.set_ylabel('Count', color='r')
                                ax2.tick_params(axis='y', labelcolor='r')
                                
                                plt.title(f"Time Series of {value_field} by {time_field}")
                                plt.tight_layout()
                                st.pyplot(fig)
                            else:
                                st.info(f"No time series data found for fields '{time_field}' and '{value_field}'.")
                        
                        except Exception as e:
                            logger.error(f"Error generating time series: {e}")
                            st.error(f"Error: {str(e)}")
                    else:
                        # Simulated time series
                        st.subheader(f"Time Series Analysis (Simulated)")
                        
                        # Create simulated time series data
                        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='W')
                        avg_values = np.random.normal(4, 0.5, size=len(dates))  # Centered around 4
                        counts = np.random.randint(5, 30, size=len(dates))
                        
                        # Create DataFrame
                        ts_df = pd.DataFrame({
                            'Date': dates,
                            'Average Value': avg_values,
                            'Count': counts
                        })
                        ts_df = ts_df.set_index('Date')
                        
                        # Display as table
                        st.dataframe(ts_df.reset_index())
                        
                        # Create chart
                        fig, ax1 = plt.subplots(figsize=(12, 6))
                        
                        # Plot average value
                        ax1.plot(ts_df.index, ts_df['Average Value'], 'b-', marker='o')
                        ax1.set_xlabel('Date')
                        ax1.set_ylabel('Average Value', color='b')
                        ax1.tick_params(axis='y', labelcolor='b')
                        
                        # Create second y-axis for count
                        ax2 = ax1.twinx()
                        ax2.plot(ts_df.index, ts_df['Count'], 'r--', marker='x')
                        ax2.set_ylabel('Count', color='r')
                        ax2.tick_params(axis='y', labelcolor='r')
                        
                        plt.title(f"Time Series of {value_field} by {time_field} (Simulated)")
                        plt.tight_layout()
                        st.pyplot(fig)
    
    # Close MongoDB connection if not in simulation mode
    if not simulation_mode and client is not None:
        client.close()

# ----- Mock Data Generation Functions -----

def generate_mock_patient_feedback(count=10):
    """Generate mock patient feedback data for simulation mode"""
    drug_names = ["Lisinopril", "Amoxicillin", "Simvastatin", "Albuterol", "Sertraline", 
                 "Metformin", "Amlodipine", "Omeprazole", "Losartan", "Hydrochlorothiazide"]
    
    comments = [
        "Very effective medication. Helped with my symptoms quickly.",
        "Experienced some side effects initially, but they went away after a few days.",
        "Not as effective as I had hoped. Will discuss with my doctor.",
        "Good value for the price. Will continue using this medication.",
        "Pharmacy service was excellent. Medication works as expected.",
        "The dosage seemed too strong for me. Had to reduce after consulting doctor.",
        "Great improvement in my condition after taking this medication.",
        "Some mild side effects but overall happy with the results.",
        "No noticeable improvement after two weeks of use.",
        "Much better than previous medication I was prescribed."
    ]
    
    feedback = []
    for i in range(count):
        # Create a feedback document
        doc = {
            "patient_id": f"P{1000 + i}",
            "drug_name": np.random.choice(drug_names),
            "rating": np.random.randint(1, 6),  # 1-5 rating
            "effectiveness": np.random.randint(1, 6),
            "side_effects": np.random.randint(1, 6),
            "comments": np.random.choice(comments),
            "timestamp": (datetime.datetime.now() - datetime.timedelta(days=np.random.randint(0, 365))).isoformat(),
            "verified_purchase": bool(np.random.binomial(1, 0.8))  # 80% are verified
        }
        feedback.append(doc)
    
    return feedback

def generate_mock_clinical_notes(count=10):
    """Generate mock clinical notes data for simulation mode"""
    drug_names = ["Lisinopril", "Amoxicillin", "Simvastatin", "Albuterol", "Sertraline"]
    conditions = ["Hypertension", "Bacterial Infection", "High Cholesterol", "Asthma", "Depression"]
    
    notes = []
    for i in range(count):
        drug_idx = np.random.randint(0, len(drug_names))
        
        # Create a clinical note document
        doc = {
            "patient_id": f"P{1000 + i}",
            "physician_id": f"DR{100 + np.random.randint(0, 20)}",
            "drug_prescribed": drug_names[drug_idx],
            "condition": conditions[drug_idx],
            "dosage": f"{np.random.choice([5, 10, 20, 25, 50, 100])} mg",
            "frequency": np.random.choice(["once daily", "twice daily", "three times daily", "as needed"]),
            "notes": f"Patient presenting with {conditions[drug_idx]}. " + 
                    f"Prescribed {drug_names[drug_idx]} for treatment. " +
                    np.random.choice([
                        "Patient responding well to treatment.",
                        "Follow-up in two weeks to assess effectiveness.",
                        "Advised patient about potential side effects.",
                        "Consider dosage adjustment if symptoms persist.",
                        "Previous medication was ineffective."
                    ]),
            "visit_date": (datetime.datetime.now() - datetime.timedelta(days=np.random.randint(0, 90))).isoformat(),
            "follow_up_recommended": bool(np.random.binomial(1, 0.7))  # 70% have follow-up
        }
        notes.append(doc)
    
    return notes

def generate_mock_drug_interactions(count=10):
    """Generate mock drug interaction data for simulation mode"""
    primary_drugs = ["Warfarin", "Simvastatin", "Fluoxetine", "Omeprazole", "Lisinopril"]
    secondary_drugs = ["Aspirin", "Ibuprofen", "Citalopram", "Clarithromycin", "Spironolactone"]
    
    interactions = []
    for i in range(count):
        primary_idx = np.random.randint(0, len(primary_drugs))
        secondary_idx = np.random.randint(0, len(secondary_drugs))
        
        # Create an interaction document
        doc = {
            "primary_drug": primary_drugs[primary_idx],
            "secondary_drug": secondary_drugs[secondary_idx],
            "severity": np.random.choice(["Minor", "Moderate", "Major"]),
            "mechanism": np.random.choice([
                "CYP450 Inhibition", 
                "P-glycoprotein Interference", 
                "Additive Effects",
                "Absorption Reduction", 
                "Metabolism Alteration"
            ]),
            "recommendation": np.random.choice([
                "Monitor closely", 
                "Adjust dosage", 
                "Avoid combination",
                "Consider alternative", 
                "No action needed"
            ]),
            "evidence_level": np.random.choice(["Strong", "Moderate", "Theoretical"]),
            "references": [f"Reference {np.random.randint(1, 100)}", f"Reference {np.random.randint(1, 100)}"],
            "last_updated": (datetime.datetime.now() - datetime.timedelta(days=np.random.randint(0, 365))).isoformat()
        }
        interactions.append(doc)
    
    return interactions

def generate_mock_adverse_events(count=10):
    """Generate mock adverse event data for simulation mode"""
    drugs = ["Atorvastatin", "Metformin", "Prednisone", "Tramadol", "Clopidogrel", 
            "Levothyroxine", "Metoprolol", "Allopurinol", "Gabapentin", "Clonazepam"]
    
    events = []
    for i in range(count):
        # Create an adverse event document
        doc = {
            "report_id": f"AE{10000 + i}",
            "patient_age": np.random.randint(18, 85),
            "patient_gender": np.random.choice(["Male", "Female"]),
            "drug_name": np.random.choice(drugs),
            "event_description": np.random.choice([
                "Skin rash and itching",
                "Nausea and vomiting",
                "Severe headache",
                "Dizziness and light-headedness",
                "Abdominal pain",
                "Fatigue and weakness",
                "Shortness of breath",
                "Muscle pain and stiffness",
                "Elevated liver enzymes",
                "Allergic reaction"
            ]),
            "onset_date": (datetime.datetime.now() - datetime.timedelta(days=np.random.randint(1, 30))).isoformat(),
            "report_date": datetime.datetime.now().isoformat(),
            "severity": np.random.choice(["Mild", "Moderate", "Severe", "Life-threatening"]),
            "outcome": np.random.choice([
                "Recovered", "Recovering", "Not recovered", "Fatal", "Unknown"
            ]),
            "causality_assessment": np.random.choice([
                "Definite", "Probable", "Possible", "Unlikely", "Unclassified"
            ]),
            "concomitant_medications": np.random.choice([True, False]),
            "reporter_type": np.random.choice(["Physician", "Pharmacist", "Patient", "Other HCP"])
        }
        events.append(doc)
    
    return events