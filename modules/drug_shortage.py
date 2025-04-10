"""
Drug Shortage Prediction module for the Pharmacy Data Science Platform.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import os
from config import logger, MODEL_DIR
from db.sqlite_db import execute_query

def show_drug_shortage():
    """Display the Drug Shortage Prediction module"""
    st.subheader("🔍 Drug Shortage Prediction")
    st.markdown("""
    This module uses machine learning to predict potential drug shortages based on 
    historical sales data, current stock levels, and market trends.
    """)
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Current Shortage Risk", "Shortage Prediction", "Historical Analysis"])
    
    # ----- Current Shortage Risk Tab -----
    with tab1:
        st.subheader("Current Stock Levels and Shortage Risk")
        
        try:
            # Get current stock data
            stock_data = execute_query("""
                SELECT 
                    d.id,
                    d.name,
                    d.category,
                    d.manufacturer,
                    d.stock,
                    AVG(s.quantity) as avg_daily_sales,
                    COUNT(DISTINCT date(s.sale_date)) as days_with_sales,
                    MAX(date(s.sale_date)) as last_sale_date
                FROM drugs d
                LEFT JOIN sales s ON d.id = s.drug_id
                GROUP BY d.id
                ORDER BY d.stock / (CASE WHEN AVG(s.quantity) IS NULL THEN 1 ELSE AVG(s.quantity) END)
            """)
            
            if stock_data.empty:
                st.warning("No drug inventory data available.")
            else:
                # Calculate days of inventory remaining
                current_date = datetime.now().date()
                stock_data['last_sale_date'] = pd.to_datetime(stock_data['last_sale_date']).dt.date
                stock_data['days_since_last_sale'] = [(current_date - date).days if pd.notna(date) else None for date in stock_data['last_sale_date']]
                
                # Fill missing values for drugs with no sales
                stock_data['avg_daily_sales'] = stock_data['avg_daily_sales'].fillna(0.1)  # Avoid division by zero
                
                # Calculate days of inventory
                stock_data['days_of_inventory'] = (stock_data['stock'] / stock_data['avg_daily_sales']).round().astype('Int64')
                
                # Classify risk
                def classify_risk(days):
                    if pd.isna(days) or days > 60:
                        return "Low"
                    elif days > 30:
                        return "Medium"
                    elif days > 14:
                        return "High"
                    else:
                        return "Critical"
                
                stock_data['shortage_risk'] = stock_data['days_of_inventory'].apply(classify_risk)
                
                # Show summary of risks
                risk_counts = stock_data['shortage_risk'].value_counts()
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    critical_count = risk_counts.get('Critical', 0)
                    st.metric("Critical Risk", critical_count, delta=None, 
                             delta_color="inverse")
                
                with col2:
                    high_count = risk_counts.get('High', 0)
                    st.metric("High Risk", high_count, delta=None, 
                             delta_color="inverse")
                
                with col3:
                    medium_count = risk_counts.get('Medium', 0)
                    st.metric("Medium Risk", medium_count, delta=None, 
                             delta_color="inverse")
                
                with col4:
                    low_count = risk_counts.get('Low', 0)
                    st.metric("Low Risk", low_count, delta=None, 
                             delta_color="inverse")
                
                # Create a color map for the risk levels
                risk_colors = {
                    'Critical': '#FF0000',  # Red
                    'High': '#FFA500',      # Orange
                    'Medium': '#FFFF00',    # Yellow
                    'Low': '#00FF00'        # Green
                }
                
                # Apply colors to the shortage_risk column
                def color_risk(val):
                    color = risk_colors.get(val, '#FFFFFF')
                    return f'background-color: {color}; color: black'
                
                # Display the data with coloring
                st.subheader("Drug Shortage Risk Assessment")
                display_cols = ['name', 'category', 'manufacturer', 'stock', 
                               'avg_daily_sales', 'days_of_inventory', 'shortage_risk']
                
                # Format and display the table
                formatted_data = stock_data[display_cols].copy()
                formatted_data.columns = ['Name', 'Category', 'Manufacturer', 'Current Stock', 
                                        'Avg Daily Sales', 'Days of Inventory', 'Shortage Risk']
                
                st.dataframe(formatted_data.style.applymap(
                    color_risk, subset=['Shortage Risk']
                ))
                
                # Visualize risk distribution
                st.subheader("Shortage Risk Distribution")
                
                fig, ax = plt.subplots(figsize=(10, 6))
                risk_order = ['Critical', 'High', 'Medium', 'Low']
                sns.countplot(x='shortage_risk', data=stock_data, order=risk_order, 
                             palette=[risk_colors[r] for r in risk_order], ax=ax)
                ax.set_xlabel('Shortage Risk')
                ax.set_ylabel('Number of Products')
                ax.set_title('Distribution of Drug Shortage Risk')
                plt.tight_layout()
                st.pyplot(fig)
                
                # Show critical and high risk items
                st.subheader("Critical and High Risk Items")
                critical_high = stock_data[stock_data['shortage_risk'].isin(['Critical', 'High'])]
                if not critical_high.empty:
                    st.dataframe(critical_high[display_cols])
                    
                    # Reorder recommendations
                    st.subheader("Recommended Reorder List")
                    reorder_df = critical_high[['name', 'stock', 'avg_daily_sales', 'days_of_inventory']].copy()
                    
                    # Calculate recommended order quantity (30 day supply)
                    reorder_df['suggested_order'] = (reorder_df['avg_daily_sales'] * 30 - reorder_df['stock']).round().astype(int)
                    reorder_df['suggested_order'] = reorder_df['suggested_order'].apply(lambda x: max(x, 0))
                    
                    reorder_df.columns = ['Product', 'Current Stock', 'Avg Daily Sales', 'Days Remaining', 'Suggested Order Qty']
                    st.dataframe(reorder_df)
                    
                    # Export reorder list
                    if st.button("Export Reorder List"):
                        csv = reorder_df.to_csv(index=False)
                        st.download_button(
                            label="Download Reorder List",
                            data=csv,
                            file_name="drug_reorder_list.csv",
                            mime="text/csv"
                        )
                else:
                    st.info("No critical or high risk items found.")
                
        except Exception as e:
            logger.error(f"Error analyzing drug shortage risk: {e}")
            st.error(f"Error analyzing shortage risk: {str(e)}")
    
    # ----- Shortage Prediction Tab -----
    with tab2:
        st.subheader("Drug Shortage Prediction Model")
        
        try:
            # Prepare historical data for training
            # This would normally use real historical shortage data
            # We'll simulate it for demonstration purposes
            
            # Get sales and stock data
            historical_data = execute_query("""
                SELECT 
                    d.id,
                    d.name,
                    d.category,
                    d.manufacturer,
                    d.stock as current_stock,
                    COUNT(DISTINCT date(s.sale_date)) as sales_days,
                    SUM(s.quantity) as total_sold,
                    AVG(s.quantity) as avg_daily_sales,
                    MAX(s.quantity) as max_daily_sales,
                    MIN(CASE WHEN s.quantity > 0 THEN s.quantity ELSE NULL END) as min_daily_sales,
                    STDDEV(s.quantity) as sales_volatility
                FROM drugs d
                LEFT JOIN sales s ON d.id = s.drug_id
                GROUP BY d.id
            """)
            
            if historical_data.empty:
                st.warning("Insufficient data for shortage prediction model.")
            else:
                # Feature engineering for demonstration
                historical_data['sales_consistency'] = historical_data['sales_volatility'] / historical_data['avg_daily_sales']
                historical_data['sales_consistency'] = historical_data['sales_consistency'].fillna(0)
                
                historical_data['days_of_inventory'] = (historical_data['current_stock'] / 
                                                       historical_data['avg_daily_sales']).fillna(999)
                
                # Simulate some shortages for demonstration
                np.random.seed(42)  # For reproducibility
                
                # Simulate shortage labels (more likely with low stock, high volatility)
                shortage_prob = 1 - historical_data['days_of_inventory'] / historical_data['days_of_inventory'].max()
                volatility_factor = historical_data['sales_consistency'] / historical_data['sales_consistency'].max()
                
                # Combine factors and normalize to 0-1
                combined_factor = (shortage_prob + volatility_factor) / 2
                combined_factor = (combined_factor - combined_factor.min()) / (combined_factor.max() - combined_factor.min())
                
                # Generate shortage events with higher probability for risk factors
                historical_data['had_shortage'] = np.random.binomial(1, combined_factor)
                
                # Prepare features and target
                features = ['avg_daily_sales', 'max_daily_sales', 'sales_volatility', 
                           'sales_consistency', 'days_of_inventory']
                
                X = historical_data[features]
                y = historical_data['had_shortage']
                
                # Standard scaling
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Model selection
                model_type = st.selectbox("Select Model Type:", ["Random Forest", "Logistic Regression"])
                
                if model_type == "Random Forest":
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                else:
                    model = LogisticRegression(random_state=42)
                
                # Train the model
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.25, random_state=42
                )
                
                model.fit(X_train, y_train)
                
                # Evaluate model
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                
                # Display metrics
                st.subheader("Model Performance")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{accuracy:.2f}")
                with col2:
                    st.metric("Precision", f"{precision:.2f}")
                with col3:
                    st.metric("Recall", f"{recall:.2f}")
                with col4:
                    st.metric("F1 Score", f"{f1:.2f}")
                
                # Confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=['No Shortage', 'Shortage'],
                           yticklabels=['No Shortage', 'Shortage'])
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.title('Confusion Matrix')
                plt.tight_layout()
                st.pyplot(plt)
                
                # Make predictions on all drugs
                all_predictions = model.predict_proba(X_scaled)[:, 1]  # Probability of shortage
                historical_data['shortage_probability'] = all_predictions
                
                # Display predictions
                st.subheader("Shortage Predictions (Next 30 Days)")
                
                prediction_df = historical_data[['name', 'category', 'current_stock', 
                                              'avg_daily_sales', 'shortage_probability']].copy()
                
                # Sort by probability
                prediction_df = prediction_df.sort_values('shortage_probability', ascending=False)
                
                # Format for display
                prediction_df['risk_level'] = pd.cut(
                    prediction_df['shortage_probability'],
                    bins=[0, 0.25, 0.5, 0.75, 1.0],
                    labels=['Low', 'Medium', 'High', 'Very High']
                )
                
                # Apply color formatting
                def color_probability(val):
                    if val < 0.25:
                        return 'background-color: #CCFFCC'  # Light green
                    elif val < 0.5:
                        return 'background-color: #FFFFCC'  # Light yellow
                    elif val < 0.75:
                        return 'background-color: #FFCC99'  # Light orange
                    else:
                        return 'background-color: #FF9999'  # Light red
                
                # Format and display prediction table
                formatted_predictions = prediction_df.copy()
                formatted_predictions['shortage_probability'] = formatted_predictions['shortage_probability'].apply(
                    lambda x: f"{x:.2%}"
                )
                formatted_predictions.columns = ['Product', 'Category', 'Current Stock', 
                                              'Avg Daily Sales', 'Shortage Probability', 'Risk Level']
                
                st.dataframe(formatted_predictions.style.applymap(
                    lambda x: color_probability(float(x.strip('%'))/100), 
                    subset=['Shortage Probability']
                ))
                
                # Save model if requested
                if st.button("Save Prediction Model"):
                    try:
                        model_dir = os.path.join(MODEL_DIR, "shortage_prediction")
                        os.makedirs(model_dir, exist_ok=True)
                        
                        # Save model and scaler
                        joblib.dump(model, os.path.join(model_dir, "shortage_model.joblib"))
                        joblib.dump(scaler, os.path.join(model_dir, "shortage_scaler.joblib"))
                        
                        st.success("Shortage prediction model saved successfully!")
                    except Exception as e:
                        logger.error(f"Error saving shortage model: {e}")
                        st.error(f"Error saving model: {str(e)}")
                
                # Feature importance for Random Forest
                if model_type == "Random Forest":
                    st.subheader("Feature Importance")
                    
                    feature_importance = pd.DataFrame({
                        'Feature': features,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    plt.figure(figsize=(10, 6))
                    sns.barplot(x='Importance', y='Feature', data=feature_importance)
                    plt.title('Feature Importance for Shortage Prediction')
                    plt.tight_layout()
                    st.pyplot(plt)
                
                # Model coefficients for Logistic Regression
                elif model_type == "Logistic Regression":
                    st.subheader("Model Coefficients")
                    
                    coef_df = pd.DataFrame({
                        'Feature': features,
                        'Coefficient': model.coef_[0]
                    }).sort_values('Coefficient', ascending=False)
                    
                    plt.figure(figsize=(10, 6))
                    sns.barplot(x='Coefficient', y='Feature', data=coef_df)
                    plt.title('Feature Coefficients for Shortage Prediction')
                    plt.tight_layout()
                    st.pyplot(plt)
        
        except Exception as e:
            logger.error(f"Error in shortage prediction: {e}")
            st.error(f"Error in shortage prediction model: {str(e)}")
    
    # ----- Historical Analysis Tab -----
    with tab3:
        st.subheader("Historical Shortage Analysis")
        
        # Since we don't have real historical shortage data, we'll create a simulation
        st.info("""
        This section simulates historical drug shortage patterns based on available data.
        In a production environment, this would use actual recorded shortage events.
        """)
        
        try:
            # Get category and monthly sales data
            category_data = execute_query("""
                SELECT 
                    d.category,
                    strftime('%Y-%m', s.sale_date) as month,
                    COUNT(DISTINCT d.id) as drug_count,
                    SUM(s.quantity) as total_sold
                FROM drugs d
                JOIN sales s ON d.id = s.drug_id
                GROUP BY d.category, month
                ORDER BY month
            """)
            
            if category_data.empty:
                st.warning("No historical data available for analysis.")
            else:
                # Process data for visualization
                category_data['month'] = pd.to_datetime(category_data['month'] + '-01')
                
                # Pivot table for category trends
                pivot_df = category_data.pivot(index='month', columns='category', values='total_sold')
                pivot_df = pivot_df.fillna(0)
                
                # Simulate shortage events
                # We'll create artificial shortage points for demonstration
                months = category_data['month'].unique()
                categories = category_data['category'].unique()
                
                shortage_data = []
                np.random.seed(42)  # For reproducibility
                
                for month in months:
                    for category in categories:
                        # Higher probability of shortage in winter for respiratory
                        month_dt = pd.to_datetime(month)
                        is_winter = month_dt.month in [12, 1, 2]
                        
                        if category == 'Respiratory' and is_winter:
                            shortage_prob = 0.4
                        elif category == 'Antibiotic' and is_winter:
                            shortage_prob = 0.3
                        else:
                            shortage_prob = 0.1
                        
                        had_shortage = np.random.binomial(1, shortage_prob)
                        
                        if had_shortage:
                            shortage_data.append({
                                'month': month,
                                'category': category,
                                'shortage_count': np.random.randint(1, 4)  # Random number of shortages
                            })
                
                shortage_df = pd.DataFrame(shortage_data)
                
                if not shortage_df.empty:
                    # Visualize historical shortages
                    st.subheader("Seasonal Shortage Patterns")
                    
                    # Pivot for heatmap
                    if len(shortage_df) > 0:
                        shortage_pivot = shortage_df.pivot_table(
                            index=shortage_df['month'].dt.strftime('%B'),  # Month name
                            columns='category',
                            values='shortage_count',
                            aggfunc='sum'
                        ).fillna(0)
                        
                        # Reorder months chronologically
                        month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                                     'July', 'August', 'September', 'October', 'November', 'December']
                        shortage_pivot = shortage_pivot.reindex(month_order)
                        
                        # Plot heatmap
                        plt.figure(figsize=(12, 8))
                        sns.heatmap(shortage_pivot, annot=True, cmap='YlOrRd', fmt='g')
                        plt.title('Shortage Events by Month and Category')
                        plt.tight_layout()
                        st.pyplot(plt)
                    
                    # Time series of shortages
                    shortage_time = shortage_df.groupby('month')['shortage_count'].sum().reset_index()
                    shortage_time = shortage_time.sort_values('month')
                    
                    # Create the time series chart
                    plt.figure(figsize=(12, 6))
                    plt.plot(shortage_time['month'], shortage_time['shortage_count'], 'r-', marker='o')
                    plt.xlabel('Month')
                    plt.ylabel('Number of Shortage Events')
                    plt.title('Drug Shortage Events Over Time')
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(plt)
                    
                    # Category breakdown
                    category_shortage = shortage_df.groupby('category')['shortage_count'].sum().reset_index()
                    category_shortage = category_shortage.sort_values('shortage_count', ascending=False)
                    
                    plt.figure(figsize=(10, 6))
                    sns.barplot(x='category', y='shortage_count', data=category_shortage)
                    plt.xlabel('Category')
                    plt.ylabel('Total Shortage Events')
                    plt.title('Drug Shortages by Category')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    st.pyplot(plt)
                    
                    # Insights
                    st.subheader("Key Insights")
                    
                    most_affected = category_shortage.iloc[0]['category']
                    winter_shortages = shortage_df[shortage_df['month'].dt.month.isin([12, 1, 2])]['shortage_count'].sum()
                    total_shortages = shortage_df['shortage_count'].sum()
                    winter_percent = (winter_shortages / total_shortages) * 100 if total_shortages > 0 else 0
                    
                    st.markdown(f"""
                    Based on the simulated historical data:
                    
                    1. **Seasonal Patterns**: {winter_percent:.1f}% of drug shortages occurred during winter months (December-February)
                    2. **Most Affected Category**: {most_affected} medications experienced the highest number of shortages
                    3. **Potential Causes**: Seasonal demand fluctuations and supply chain disruptions appear to be key factors
                    
                    **Recommendations:**
                    - Increase safety stock for {most_affected} products before winter season
                    - Develop alternative supplier relationships for high-risk categories
                    - Implement early warning monitoring for seasonal products
                    """)
                else:
                    st.info("No historical shortage events to analyze.")
        
        except Exception as e:
            logger.error(f"Error in historical shortage analysis: {e}")
            st.error(f"Error analyzing historical shortages: {str(e)}")