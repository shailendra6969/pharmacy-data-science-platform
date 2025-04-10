"""
Data loading utilities for the Pharmacy Data Science Platform.
"""
import pandas as pd
import os
import streamlit as st
from config import CSV_PATH, logger

def load_csv_data():
    """
    Load data from the sample drugs CSV file.
    If the file doesn't exist, create a placeholder.
    """
    try:
        # Check if the file exists
        if os.path.exists(CSV_PATH):
            df = pd.read_csv(CSV_PATH)
            logger.info(f"Loaded {len(df)} records from {CSV_PATH}")
            return df
        else:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
            
            # If file doesn't exist, create a sample dataset
            logger.warning(f"CSV file not found at {CSV_PATH}. Creating sample data.")
            
            # Create sample drug data
            categories = ['Cardiovascular', 'Antibiotic', 'Antidiabetic', 'Respiratory', 
                         'Psychiatric', 'Gastrointestinal', 'Pain Management', 'Hormonal',
                         'Allergy', 'Dermatology']
            
            manufacturers = ['PharmaCorp', 'MediGen', 'BioHealth', 'CureAll', 'LifeScience',
                            'MedTech', 'HealthWay', 'VitaCorp', 'GeneriMed', 'PharmaPlus']
            
            sample_data = []
            for i in range(1, 101):
                category_idx = (i - 1) % len(categories)
                manufacturer_idx = (i - 1) % len(manufacturers)
                
                drug = {
                    'id': i,
                    'name': f"Drug-{i:03d}",
                    'category': categories[category_idx],
                    'price': round(100 + (i % 10) * 50 + (i % 3) * 25, 2),
                    'dosage': f"{(i % 3) + 1} per day",
                    'description': f"Sample description for Drug-{i:03d}",
                    'manufacturer': manufacturers[manufacturer_idx],
                    'stock': (i % 5 + 3) * 100
                }
                sample_data.append(drug)
            
            df = pd.DataFrame(sample_data)
            
            # Save the sample dataset
            df.to_csv(CSV_PATH, index=False)
            logger.info(f"Created and saved sample data with {len(df)} records to {CSV_PATH}")
            return df
    
    except Exception as e:
        logger.error(f"Error loading CSV data: {e}")
        st.error(f"Error loading data: {str(e)}")
        # Return an empty DataFrame as fallback
        return pd.DataFrame()

def load_uploaded_data(uploaded_file, required_columns=None):
    """
    Load data from an uploaded file with validation.
    
    Args:
        uploaded_file: The file uploaded via Streamlit
        required_columns: List of column names that must be present
        
    Returns:
        DataFrame or None if validation fails
    """
    try:
        # Detect file type and read accordingly
        file_name = uploaded_file.name
        if file_name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif file_name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None
        
        # Validate required columns if specified
        if required_columns:
            missing = [col for col in required_columns if col not in df.columns]
            if missing:
                st.error(f"Missing required columns: {', '.join(missing)}")
                return None
                
        return df
        
    except Exception as e:
        logger.error(f"Error loading uploaded file: {e}")
        st.error(f"Error reading file: {str(e)}")
        return None