"""
Drug Price Prediction module for the Pharmacy Data Science Platform.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import sqlite3
from config import logger, MODEL_DIR
from utils.data_loader import load_uploaded_data
from db.sqlite_db import execute_query

def load_or_train_price_model(X, y):
    """
    Load existing model or train a new one
    
    Args:
        X: Feature DataFrame
        y: Target Series
        
    Returns:
        Trained model
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "drug_price_model.joblib")
    
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            logger.info("Loaded existing drug price prediction model")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            # If loading fails, train a new model
    
    logger.info("Training new drug price prediction model")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    pipeline.fit(X, y)
    
    try:
        joblib.dump(pipeline, model_path)
        logger.info("Saved new drug price prediction model")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
    
    return pipeline

def evaluate_model(model, X, y):
    """
    Evaluate model performance
    
    Args:
        model: Trained model
        X: Feature DataFrame
        y: Target Series
        
    Returns:
        Dictionary of evaluation metrics
    """
    y_pred = model.predict(X)
    
    metrics = {
        'MAE': mean_absolute_error(y, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y, y_pred)),
        'R²': r2_score(y, y_pred)
    }
    
    return metrics, y_pred

def show_price_prediction():
    """Display the Drug Price Prediction module"""
    st.subheader("💰 Drug Price Prediction (ML Model)")
    st.markdown("""
    This module uses a Random Forest Regression model to predict drug prices based on features like
    manufacturing complexity, R&D cost, patent status, and market competition.
    """)
    
    tab1, tab2, tab3 = st.tabs(["Predict Prices", "Train Model", "Model Performance"])
    
    # ------- Predict Prices Tab -------
    with tab1:
        st.subheader("Predict Drug Price")
        
        col1, col2 = st.columns(2)
        with col1:
            form_complexity = st.slider("Manufacturing Complexity (1-10):", 1, 10, 5)
            r_and_d_cost = st.number_input("R&D Cost (₹ Thousands):", 100, 5000, 1000, step=100)
            patent_expiry = st.slider("Years Until Patent Expiry:", 0, 20, 10)
            active_ingredients = st.slider("Number of Active Ingredients:", 1, 10, 2)
        
        with col2:
            try:
                # Get categories from database
                categories_df = execute_query("SELECT DISTINCT category FROM drugs ORDER BY category")
                if not categories_df.empty:
                    categories = categories_df['category'].tolist()
                else:
                    categories = ["Cardiovascular", "Antibiotic", "Antidiabetic", "Respiratory", 
                                 "Psychiatric", "Gastrointestinal", "Pain Management", "Hormonal"]
            except Exception as e:
                categories = ["Cardiovascular", "Antibiotic", "Antidiabetic", "Respiratory", 
                             "Psychiatric", "Gastrointestinal", "Pain Management", "Hormonal"]
                logger.error(f"Error fetching categories: {e}")
            
            category = st.selectbox("Treatment Category:", categories)
            competition = st.slider("Market Competition (1-10):", 1, 10, 5, 
                                  help="Higher values indicate more competition")
            dosage_units = st.number_input("Dosage Units per Package:", 10, 500, 30, step=10)
        
        # Load sample data for initial model training if needed
        try:
            df = execute_query("SELECT category, price FROM drugs")
            if df.empty:
                st.error("No drug data available in the database.")
                X_sample = pd.DataFrame({'dummy': [0]})
                y_sample = pd.Series([0])
            else:
                X_sample = pd.get_dummies(df['category'])
                y_sample = df['price']
                
                # Create a message about data used for training
                st.info(f"Model trained on {len(y_sample)} drug records from the database.")
        except Exception as e:
            logger.error(f"Error loading sample data: {e}")
            st.error(f"Error loading training data: {str(e)}")
            X_sample = pd.DataFrame({'dummy': [0]})
            y_sample = pd.Series([0])
        
        # Load or train model
        try:
            model = load_or_train_price_model(X_sample, y_sample)
        except Exception as e:
            logger.error(f"Error in model preparation: {e}")
            st.error(f"Error preparing prediction model: {str(e)}")
            model = None
        
        if model is not None and st.button("Predict Price"):
            try:
                # Prepare input data
                input_data = pd.DataFrame({
                    'complexity': [form_complexity],
                    'r_and_d_cost': [r_and_d_cost],
                    'patent_expiry': [patent_expiry],
                    'active_ingredients': [active_ingredients],
                    'competition': [competition],
                    'dosage_units': [dosage_units]
                })
                
                # One-hot encode category
                category_encoded = pd.get_dummies(pd.Series([category]), prefix='category')
                
                # Ensure all columns from training are present
                for col in X_sample.columns:
                    if col not in category_encoded.columns:
                        category_encoded[col] = 0
                
                # Select only columns that were in the training data
                category_encoded = category_encoded[X_sample.columns]
                
                # Combine numerical and categorical features
                input_data = pd.concat([input_data, category_encoded], axis=1)
                
                # Make prediction
                prediction = model.predict(input_data)
                
                # Display result
                st.success(f"Predicted Price: ₹{prediction[0]:,.2f}")
                
                # Show feature importance if possible
                if hasattr(model, 'named_steps') and hasattr(model.named_steps['model'], 'feature_importances_'):
                    st.subheader("Feature Importance")
                    feature_importance = model.named_steps['model'].feature_importances_
                    features = input_data.columns
                    
                    importance_df = pd.DataFrame({
                        'Feature': features,
                        'Importance': feature_importance
                    }).sort_values('Importance', ascending=False)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.barh(importance_df['Feature'], importance_df['Importance'])
                    ax.set_xlabel('Importance')
                    ax.set_title('Feature Importance')
                    plt.tight_layout()
                    st.pyplot(fig)
            
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                st.error(f"Error making prediction: {str(e)}")
    
    # ------- Train Model Tab -------
    with tab2:
        st.subheader("Train New Price Prediction Model")
        st.info("""
        Upload a CSV file with drug features and prices to train a custom model.
        The CSV must contain at least a 'price' column and feature columns.
        """)
        
        uploaded_file = st.file_uploader("Upload CSV with drug features and prices:", type=['csv'])
        
        if uploaded_file:
            df = load_uploaded_data(uploaded_file, required_columns=['price'])
            
            if df is not None:
                st.success("Data uploaded successfully")
                st.dataframe(df.head())
                
                # Separate features and target
                if 'price' in df.columns:
                    X = df.drop('price', axis=1)
                    
                    # Handle categorical features
                    categorical_cols = X.select_dtypes(include=['object']).columns
                    if not categorical_cols.empty:
                        X = pd.get_dummies(X, columns=categorical_cols)
                    
                    y = df['price']
                    
                    # Split data for training and testing
                    test_size = st.slider("Test Set Size (%):", 10, 50, 20) / 100
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42
                    )
                    
                    st.info(f"Training data: {X_train.shape[0]} samples, Test data: {X_test.shape[0]} samples")
                    
                    # Model parameters
                    col1, col2 = st.columns(2)
                    with col1:
                        n_estimators = st.slider("Number of Trees:", 50, 500, 100, step=10)
                    with col2:
                        max_features = st.select_slider("Max Features:", 
                                                     options=["auto", "sqrt", "log2", None], 
                                                     value="auto")
                    
                    if st.button("Train Model"):
                        with st.spinner("Training model..."):
                            # Configure model with user parameters
                            pipeline = Pipeline([
                                ('scaler', StandardScaler()),
                                ('model', RandomForestRegressor(
                                    n_estimators=n_estimators,
                                    max_features=max_features,
                                    random_state=42
                                ))
                            ])
                            
                            # Train model
                            pipeline.fit(X_train, y_train)
                            
                            # Evaluate on test set
                            metrics, y_pred = evaluate_model(pipeline, X_test, y_test)
                            
                            # Display metrics
                            st.subheader("Model Performance")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Mean Absolute Error", f"₹{metrics['MAE']:.2f}")
                            with col2:
                                st.metric("Root Mean Squared Error", f"₹{metrics['RMSE']:.2f}")
                            with col3:
                                st.metric("R² Score", f"{metrics['R²']:.4f}")
                            
                            # Save model
                            model_path = os.path.join(MODEL_DIR, "drug_price_model.joblib")
                            joblib.dump(pipeline, model_path)
                            st.success(f"Model trained and saved to {model_path}")
                            
                            # Feature importance
                            if hasattr(pipeline.named_steps['model'], 'feature_importances_'):
                                st.subheader("Feature Importance")
                                feature_importance = pipeline.named_steps['model'].feature_importances_
                                features = X.columns
                                
                                importance_df = pd.DataFrame({
                                    'Feature': features,
                                    'Importance': feature_importance
                                }).sort_values('Importance', ascending=False)
                                
                                fig, ax = plt.subplots(figsize=(10, 6))
                                ax.barh(importance_df['Feature'][:15], importance_df['Importance'][:15])
                                ax.set_xlabel('Importance')
                                ax.set_title('Top 15 Feature Importance')
                                plt.tight_layout()
                                st.pyplot(fig)
                else:
                    st.error("CSV must contain a 'price' column")
    
    # ------- Model Performance Tab -------
    with tab3:
        st.subheader("Current Model Performance")
        
        # Try to load existing model
        model_path = os.path.join(MODEL_DIR, "drug_price_model.joblib")
        if os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
                
                # Get some data to evaluate
                try:
                    df = execute_query("""
                        SELECT d.category, d.price, d.manufacturer, d.dosage
                        FROM drugs d
                    """)
                    
                    if not df.empty:
                        # Prepare features
                        X = pd.get_dummies(df.drop('price', axis=1))
                        y = df['price']
                        
                        # Evaluate model
                        metrics, y_pred = evaluate_model(model, X, y)
                        
                        # Display metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Mean Absolute Error", f"₹{metrics['MAE']:.2f}")
                        with col2:
                            st.metric("Root Mean Squared Error", f"₹{metrics['RMSE']:.2f}")
                        with col3:
                            st.metric("R² Score", f"{metrics['R²']:.4f}")
                        
                        # Plot actual vs predicted
                        st.subheader("Actual vs Predicted Prices")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.scatter(y, y_pred, alpha=0.5)
                        
                        # Add perfect prediction line
                        min_val = min(y.min(), y_pred.min())
                        max_val = max(y.max(), y_pred.max())
                        ax.plot([min_val, max_val], [min_val, max_val], 'r--')
                        
                        ax.set_xlabel('Actual Price')
                        ax.set_ylabel('Predicted Price')
                        ax.set_title('Actual vs Predicted Drug Prices')
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Show residuals
                        st.subheader("Prediction Residuals")
                        residuals = y - y_pred
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.scatter(y_pred, residuals, alpha=0.5)
                        ax.axhline(y=0, color='r', linestyle='--')
                        ax.set_xlabel('Predicted Price')
                        ax.set_ylabel('Residual')
                        ax.set_title('Residual Plot')
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.warning("No drug data available to evaluate model performance.")
                
                except Exception as e:
                    logger.error(f"Error evaluating model: {e}")
                    st.error(f"Error evaluating model performance: {str(e)}")
            
            except Exception as e:
                logger.error(f"Error loading model for evaluation: {e}")
                st.error(f"Error loading model: {str(e)}")
        else:
            st.info("No trained model found. Please train a model first in the 'Train Model' tab.")