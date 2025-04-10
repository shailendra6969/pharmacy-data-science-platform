"""
Drug price prediction model for the Pharmacy Data Science Platform.
"""
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import logging
import matplotlib.pyplot as plt

from config import MODEL_DIR, logger

def load_or_train_price_model(X, y, force_retrain=False):
    """
    Load existing price prediction model or train a new one
    
    Parameters:
    -----------
    X : DataFrame
        Features for training
    y : Series
        Target values (prices)
    force_retrain : bool
        Whether to force retraining even if a model exists
        
    Returns:
    --------
    model : sklearn Pipeline
        Trained model pipeline
    """
    model_path = os.path.join(MODEL_DIR, "drug_price_model.joblib")
    
    if os.path.exists(model_path) and not force_retrain:
        # Load existing model
        try:
            model = joblib.load(model_path)
            logger.info("Loaded existing drug price prediction model")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.info("Training new model instead")
    
    # Train new model
    logger.info("Training new drug price prediction model")
    
    try:
        # Create a pipeline with preprocessing and model
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        
        # Train the model
        pipeline.fit(X, y)
        
        # Save the model
        joblib.dump(pipeline, model_path)
        logger.info("Saved new drug price prediction model")
        
        return pipeline
    except Exception as e:
        logger.error(f"Error training price model: {e}")
        return None

def evaluate_price_model(model, X, y):
    """
    Evaluate the price prediction model
    
    Parameters:
    -----------
    model : sklearn Pipeline
        Trained model to evaluate
    X : DataFrame
        Features for evaluation
    y : Series
        True target values
        
    Returns:
    --------
    dict
        Dictionary of evaluation metrics
    """
    if model is None:
        logger.error("No model provided for evaluation")
        return {}
    
    try:
        # Split data for evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        }
        
        logger.info(f"Model evaluation metrics: {metrics}")
        return metrics
    except Exception as e:
        logger.error(f"Error evaluating price model: {e}")
        return {}

def predict_drug_price(model, features):
    """
    Predict drug price using the trained model
    
    Parameters:
    -----------
    model : sklearn Pipeline
        Trained price prediction model
    features : dict or DataFrame
        Features for prediction
        
    Returns:
    --------
    float
        Predicted price
    """
    if model is None:
        logger.error("No model provided for prediction")
        return None
    
    try:
        # Convert features to DataFrame if it's a dict
        if isinstance(features, dict):
            features_df = pd.DataFrame([features])
        else:
            features_df = features
            
        # Make prediction
        predicted_price = model.predict(features_df)[0]
        
        logger.info(f"Predicted drug price: {predicted_price:.2f}")
        return predicted_price
    except Exception as e:
        logger.error(f"Error predicting drug price: {e}")
        return None

def get_feature_importance(model):
    """
    Get feature importance from the trained model
    
    Parameters:
    -----------
    model : sklearn Pipeline
        Trained price prediction model
        
    Returns:
    --------
    DataFrame
        DataFrame with feature names and importance scores
    """
    if model is None:
        logger.error("No model provided for feature importance")
        return pd.DataFrame()
    
    try:
        # Extract feature importance from the model
        rf_model = model.named_steps['model']
        feature_importance = rf_model.feature_importances_
        
        # Get feature names from the model
        if hasattr(model, 'feature_names_in_'):
            feature_names = model.feature_names_in_
        else:
            feature_names = [f"Feature {i}" for i in range(len(feature_importance))]
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        return importance_df
    except Exception as e:
        logger.error(f"Error getting feature importance: {e}")
        return pd.DataFrame()

def plot_feature_importance(importance_df, figsize=(10, 6)):
    """
    Create a feature importance plot
    
    Parameters:
    -----------
    importance_df : DataFrame
        DataFrame with feature importance data
    figsize : tuple
        Figure size as (width, height)
        
    Returns:
    --------
    fig, ax
        Matplotlib figure and axis objects
    """
    if importance_df.empty:
        logger.error("No feature importance data provided")
        return None, None
    
    try:
        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(importance_df['Feature'], importance_df['Importance'])
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importance for Price Prediction')
        plt.tight_layout()
        
        return fig, ax
    except Exception as e:
        logger.error(f"Error plotting feature importance: {e}")
        return None, None  # ← this uses standard spaces

 