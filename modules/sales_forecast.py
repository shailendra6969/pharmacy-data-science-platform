"""
Sales Forecast module for the Pharmacy Data Science Platform.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import sqlite3
from config import logger, DB_PATH
from utils.data_loader import load_uploaded_data
from db.sqlite_db import execute_query

def forecast_sales(df, steps, order=(5,1,0)):
    """
    Forecast future sales using ARIMA model
    
    Args:
        df: DataFrame with 'Date' and 'Sales' columns
        steps: Number of time periods to forecast
        order: ARIMA model order (p,d,q)
        
    Returns:
        Tuple of (original time series, forecast, confidence intervals)
    """
    try:
        # Check for required columns
        if 'Date' not in df.columns or 'Sales' not in df.columns:
            st.error("Data must contain 'Date' and 'Sales' columns")
            return None, None, None
            
        # Convert date and set as index
        if not pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Date'] = pd.to_datetime(df['Date'])
        
        df.set_index('Date', inplace=True)
        ts = df['Sales']
        
        # Check if enough data points
        if len(ts) < 10:
            st.warning("Not enough data points for reliable forecasting. Need at least 10 data points.")
            return ts, None, None
            
        # Fit ARIMA model
        model = ARIMA(ts, order=order)
        model_fit = model.fit()
        
        # Generate forecast
        forecast_results = model_fit.get_forecast(steps=steps)
        forecast = forecast_results.predicted_mean
        conf_int = forecast_results.conf_int()
        
        return ts, forecast, conf_int, model_fit
        
    except Exception as e:
        logger.error(f"Error in sales forecasting: {e}")
        st.error(f"Error in forecasting: {str(e)}")
        return None, None, None, None

def show_sales_forecast():
    """Display the Sales Forecast module"""
    st.subheader("📈 Drug Sales Forecast")
    st.markdown("""
    This module uses ARIMA models to forecast future drug sales based on historical data.
    You can upload your own CSV file with 'Date' and 'Sales' columns or use sample data from the database.
    """)
    
    # Data source selection
    data_source = st.radio("Select Data Source:", 
                         ["Sample Database Data", "Upload CSV", "Filter by Drug Category"])
    
    data = None
    
    if data_source == "Sample Database Data":
        try:
            # Get all sales data
            sample_data = execute_query("""
                SELECT date(sale_date) as Date, SUM(total_price) as Sales
                FROM sales
                GROUP BY date(sale_date)
                ORDER BY Date
            """)
            
            if sample_data.empty:
                st.error("No sales data available in the database.")
            else:
                st.success("Using sample sales data from database")
                st.subheader("Sample Data Preview")
                st.dataframe(sample_data.head())
                data = sample_data
        
        except Exception as e:
            logger.error(f"Error loading sample data: {e}")
            st.error(f"Error loading sample data: {str(e)}")
            
    elif data_source == "Upload CSV":
        uploaded = st.file_uploader("Upload Sales CSV with Date and Sales columns:", type=['csv'])
        if uploaded:
            data = load_uploaded_data(uploaded, required_columns=['Date', 'Sales'])
            if data is not None:
                st.success("Data uploaded successfully")
                st.dataframe(data.head())
    
    elif data_source == "Filter by Drug Category":
        try:
            # Get all categories
            categories = execute_query("SELECT DISTINCT category FROM drugs ORDER BY category")
            
            if categories.empty:
                st.error("No category data available in the database.")
            else:
                category_list = categories['category'].tolist()
                selected_category = st.selectbox("Select Drug Category:", category_list)
                
                # Get sales data for selected category
                category_data = execute_query(f"""
                    SELECT date(s.sale_date) as Date, SUM(s.total_price) as Sales
                    FROM sales s
                    JOIN drugs d ON s.drug_id = d.id
                    WHERE d.category = ?
                    GROUP BY date(s.sale_date)
                    ORDER BY Date
                """, params=(selected_category,))
                
                if category_data.empty:
                    st.error(f"No sales data available for category: {selected_category}")
                else:
                    st.success(f"Loaded sales data for category: {selected_category}")
                    st.dataframe(category_data.head())
                    data = category_data
        
        except Exception as e:
            logger.error(f"Error loading category data: {e}")
            st.error(f"Error loading category data: {str(e)}")
    
    if data is not None:
        # Model parameters
        st.subheader("Model Parameters")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            p = st.slider("AR order (p):", 0, 10, 5, 
                         help="Autoregressive order - number of lag observations in the model")
        with col2:
            d = st.slider("Differencing order (d):", 0, 2, 1,
                         help="Integrated order - number of times to difference the data")
        with col3:
            q = st.slider("MA order (q):", 0, 10, 0,
                         help="Moving Average order - size of the moving average window")
        
        # Forecast horizon
        period = st.slider("Forecast horizon (days):", 5, 60, 15,
                         help="Number of days to forecast into the future")
        
        # Generate forecast button
        if st.button("Generate Forecast"):
            with st.spinner("Generating forecast..."):
                ts, forecast, conf_int, model_fit = forecast_sales(data, period, order=(p, d, q))
                
                if ts is not None:
                    # Show historical data
                    st.subheader("Historical Sales Data")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(ts.index, ts.values)
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Sales')
                    ax.set_title('Historical Sales')
                    ax.grid(True)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Seasonal Decomposition
                    if len(ts) >= 14:  # Need enough data points for decomposition
                        st.subheader("Seasonal Decomposition")
                        try:
                            # Try different seasonal periods
                            if len(ts) >= 365:  # If we have a year of data
                                period = 365  # Annual seasonality
                            elif len(ts) >= 30:
                                period = 30  # Monthly seasonality
                            else:
                                period = 7  # Weekly seasonality
                                
                            decomposition = seasonal_decompose(ts, model='additive', period=period)
                            
                            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 12))
                            decomposition.observed.plot(ax=ax1)
                            ax1.set_title('Observed')
                            ax1.set_xlabel('')
                            
                            decomposition.trend.plot(ax=ax2)
                            ax2.set_title('Trend')
                            ax2.set_xlabel('')
                            
                            decomposition.seasonal.plot(ax=ax3)
                            ax3.set_title('Seasonality')
                            ax3.set_xlabel('')
                            
                            decomposition.resid.plot(ax=ax4)
                            ax4.set_title('Residuals')
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                        except Exception as e:
                            st.warning(f"Could not perform seasonal decomposition: {str(e)}")
                    
                    # Show forecast if available
                    if forecast is not None:
                        forecast_index = pd.date_range(start=ts.index[-1] + pd.Timedelta(days=1), periods=len(forecast))
                        
                        st.subheader("Sales Forecast")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        # Plot historical data
                        ax.plot(ts.index, ts.values, label='Historical')
                        
                        # Plot forecast
                        ax.plot(forecast_index, forecast.values, color='red', label='Forecast')
                        
                        # Plot confidence intervals if available
                        if conf_int is not None:
                            ax.fill_between(forecast_index, 
                                           conf_int.iloc[:, 0].values, 
                                           conf_int.iloc[:, 1].values,
                                           color='pink', alpha=0.3, 
                                           label='95% Confidence Interval')
                        
                        ax.set_xlabel('Date')
                        ax.set_ylabel('Sales')
                        ax.set_title('Sales Forecast with ARIMA')
                        ax.legend()
                        ax.grid(True)
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Display forecast data in table
                        st.subheader("Forecast Data")
                        forecast_df = pd.DataFrame({
                            'Date': forecast_index,
                            'Forecast': forecast.values,
                            'Lower CI': conf_int.iloc[:, 0].values,
                            'Upper CI': conf_int.iloc[:, 1].values
                        })
                        st.dataframe(forecast_df)
                        
                        # Download forecast as CSV
                        csv = forecast_df.to_csv(index=False)
                        st.download_button(
                            label="Download Forecast CSV",
                            data=csv,
                            file_name="sales_forecast.csv",
                            mime="text/csv"
                        )
                        
                        # Show model summary
                        if model_fit is not None:
                            with st.expander("Show Model Summary"):
                                st.text(str(model_fit.summary()))