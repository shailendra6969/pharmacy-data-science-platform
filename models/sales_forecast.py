"""
Sales forecasting models for the Pharmacy Data Science Platform.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import logging
import streamlit as st

from config import logger

def forecast_sales(df, steps=30, order=(5,1,0)):
    """
    Forecast future sales using ARIMA model
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame containing 'Date' and 'Sales' columns
    steps : int
        Number of steps to forecast
    order : tuple
        ARIMA model order (p,d,q)
        
    Returns:
    --------
    original_ts : Series
        Original time series
    forecast : Series
        Forecasted values
    conf_int : DataFrame
        Confidence intervals for forecast
    """
    try:
        # Validate input data
        if 'Date' not in df.columns or 'Sales' not in df.columns:
            logger.error("CSV must contain 'Date' and 'Sales' columns")
            return None, None, None
            
        # Convert date and set as index
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        original_ts = df['Sales']
        
        # Check for enough data points
        if len(original_ts) < 10:
            logger.warning("Not enough data points for reliable forecasting. Need at least 10 data points.")
            return original_ts, None, None
            
        # Fit ARIMA model
        model = ARIMA(original_ts, order=order)
        model_fit = model.fit()
        
        # Generate forecast with confidence intervals
        forecast_results = model_fit.get_forecast(steps=steps)
        forecast = forecast_results.predicted_mean
        conf_int = forecast_results.conf_int()
        
        logger.info(f"Generated forecast for {steps} steps with ARIMA{order}")
        return original_ts, forecast, conf_int
    except Exception as e:
        logger.error(f"Error in sales forecasting: {e}")
        if st:  # Check if streamlit is being used
            st.error(f"Error in forecasting: {str(e)}")
        return None, None, None

def plot_forecast(original_ts, forecast, conf_int, figsize=(10, 6)):
    """
    Plot the sales forecast with confidence intervals
    
    Parameters:
    -----------
    original_ts : Series
        Original time series
    forecast : Series
        Forecasted values
    conf_int : DataFrame
        Confidence intervals for forecast
    figsize : tuple
        Figure size as (width, height)
        
    Returns:
    --------
    fig, ax
        Matplotlib figure and axis objects
    """
    if original_ts is None:
        logger.error("No original time series provided")
        return None, None
        
    try:
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot historical data
        ax.plot(original_ts.index, original_ts.values, label='Historical')
        
        if forecast is not None:
            # Create a date range for the forecast period
            last_date = original_ts.index[-1]
            forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(forecast))
            
            # Plot forecast
            ax.plot(forecast_index, forecast.values, color='red', label='Forecast')
            
            # Plot confidence intervals
            if conf_int is not None:
                ax.fill_between(forecast_index, 
                              conf_int.iloc[:, 0].values, 
                              conf_int.iloc[:, 1].values,
                              color='pink', alpha=0.3, label='95% Confidence Interval')
            
        ax.set_xlabel('Date')
        ax.set_ylabel('Sales')
        ax.set_title('Sales Forecast with ARIMA')
        ax.legend()
        ax.grid(True)
        
        return fig, ax
    except Exception as e:
        logger.error(f"Error plotting forecast: {e}")
        return None, None

def perform_seasonal_decomposition(ts, period=7):
    """
    Perform seasonal decomposition of time series
    
    Parameters:
    -----------
    ts : Series
        Time series to decompose
    period : int
        Period for seasonal decomposition
        
    Returns:
    --------
    decomposition : statsmodels DecomposeResult
        Decomposition result object
    """
    try:
        # Check for enough data points
        if len(ts) < period * 2:
            logger.warning(f"Not enough data points for seasonal decomposition with period {period}")
            return None
            
        # Perform decomposition
        decomposition = seasonal_decompose(ts, period=period)
        
        logger.info(f"Performed seasonal decomposition with period {period}")
        return decomposition
    except Exception as e:
        logger.error(f"Error in seasonal decomposition: {e}")
        return None

def plot_seasonal_decomposition(decomposition, figsize=(12, 10)):
    """
    Plot the seasonal decomposition
    
    Parameters:
    -----------
    decomposition : statsmodels DecomposeResult
        Decomposition result object
    figsize : tuple
        Figure size as (width, height)
        
    Returns:
    --------
    fig
        Matplotlib figure object
    """
    if decomposition is None:
        logger.error("No decomposition result provided")
        return None
        
    try:
        fig = plt.figure(figsize=figsize)
        
        # Plot the decomposition
        ax1 = fig.add_subplot(411)
        ax1.plot(decomposition.observed)
        ax1.set_title('Observed')
        ax1.set_ylabel('Sales')
        
        ax2 = fig.add_subplot(412)
        ax2.plot(decomposition.trend)
        ax2.set_title('Trend')
        ax2.set_ylabel('Sales')
        
        ax3 = fig.add_subplot(413)
        ax3.plot(decomposition.seasonal)
        ax3.set_title('Seasonality')
        ax3.set_ylabel('Sales')
        
        ax4 = fig.add_subplot(414)
        ax4.plot(decomposition.resid)
        ax4.set_title('Residuals')
        ax4.set_ylabel('Sales')
        
        plt.tight_layout()
        
        return fig
    except Exception as e:
        logger.error(f"Error plotting seasonal decomposition: {e}")
        return None

def calculate_forecast_metrics(forecast, original_ts=None):
    """
    Calculate summary metrics for the forecast
    
    Parameters:
    -----------
    forecast : Series
        Forecasted values
    original_ts : Series or None
        Original time series for comparison
        
    Returns:
    --------
    dict
        Dictionary of forecast metrics
    """
    if forecast is None:
        logger.error("No forecast provided")
        return {}
        
    try:
        metrics = {}
        
        # Basic forecast statistics
        metrics['average'] = forecast.mean()
        metrics['min'] = forecast.min()
        metrics['max'] = forecast.max()
        
        # Trend analysis
        metrics['trend'] = "Increasing" if forecast.iloc[-1] > forecast.iloc[0] else "Decreasing"
        metrics['percent_change'] = ((forecast.iloc[-1] - forecast.iloc[0]) / forecast.iloc[0]) * 100
        
        # Compare with historical data if provided
        if original_ts is not None:
            metrics['historical_avg'] = original_ts.mean()
            metrics['historical_std'] = original_ts.std()
            metrics['forecast_vs_historical'] = forecast.mean() / original_ts.mean() * 100 - 100
        
        logger.info(f"Calculated forecast metrics: {metrics}")
        return metrics
    except Exception as e:
        logger.error(f"Error calculating forecast metrics: {e}")
        return {}
