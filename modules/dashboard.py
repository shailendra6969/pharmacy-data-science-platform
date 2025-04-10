"""
Dashboard module for the Pharmacy Data Science Platform.
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
from config import logger
from db.sqlite_db import get_db_connection, execute_query

def show_dashboard():
    """Display the main dashboard with key metrics and charts"""
    st.subheader("📊 Pharmacy Analytics Dashboard")
    
    try:
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_drugs = execute_query("SELECT COUNT(*) as count FROM drugs").iloc[0,0]
            st.metric("Total Products", total_drugs)
            
        with col2:
            total_sales = execute_query("SELECT SUM(total_price) as total FROM sales").iloc[0,0]
            st.metric("Total Sales", f"₹{total_sales:,.2f}")
            
        with col3:
            popular_drug = execute_query("""
                SELECT d.name, SUM(s.quantity) as total_sold 
                FROM sales s JOIN drugs d ON s.drug_id = d.id 
                GROUP BY d.name ORDER BY total_sold DESC LIMIT 1
            """)
            st.metric("Most Popular Drug", popular_drug.iloc[0,0])
            
        with col4:
            low_stock = execute_query("SELECT COUNT(*) as count FROM drugs WHERE stock < 400").iloc[0,0]
            st.metric("Low Stock Items", low_stock)
        
        # Time period filter for charts
        st.subheader("Filter Data")
        time_period = st.radio("Select Time Period:", 
                              ["Last 30 Days", "Last 90 Days", "Last 180 Days", "All Time"],
                              horizontal=True)
        
        if time_period == "Last 30 Days":
            date_filter = "AND date(sale_date) >= date('now', '-30 days')"
        elif time_period == "Last 90 Days":
            date_filter = "AND date(sale_date) >= date('now', '-90 days')"
        elif time_period == "Last 180 Days":
            date_filter = "AND date(sale_date) >= date('now', '-180 days')"
        else:
            date_filter = ""
        
        # Sales Trends Chart
        st.subheader("Sales Trends")
        sales_data = execute_query(f"""
            SELECT date(sale_date) as date, SUM(total_price) as daily_sales
            FROM sales
            WHERE 1=1 {date_filter}
            GROUP BY date
            ORDER BY date
        """)
        
        if not sales_data.empty:
            sales_data['date'] = pd.to_datetime(sales_data['date'])
            sales_data = sales_data.set_index('date')
            sales_data['7day_avg'] = sales_data['daily_sales'].rolling(window=7).mean()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(sales_data.index, sales_data['daily_sales'], 'b-', alpha=0.5, label='Daily Sales')
            ax.plot(sales_data.index, sales_data['7day_avg'], 'r-', label='7-Day Average')
            ax.set_xlabel('Date')
            ax.set_ylabel('Sales (₹)')
            ax.set_title('Daily Sales with 7-Day Moving Average')
            ax.legend()
            fig.tight_layout()
            st.pyplot(fig)
        else:
            st.info("No sales data available for the selected time period.")
        
        # Top Products by Revenue
        st.subheader("Top Selling Products")
        top_drugs = execute_query(f"""
            SELECT d.name, SUM(s.quantity) as total_sold, SUM(s.total_price) as total_revenue
            FROM sales s JOIN drugs d ON s.drug_id = d.id
            WHERE 1=1 {date_filter}
            GROUP BY d.name
            ORDER BY total_revenue DESC
            LIMIT 5
        """)
        
        if not top_drugs.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(top_drugs['name'], top_drugs['total_revenue'])
            ax.set_xlabel('Drug Name')
            ax.set_ylabel('Revenue (₹)')
            ax.set_title('Top 5 Revenue-Generating Drugs')
            plt.xticks(rotation=45, ha='right')
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 5000,
                        f'₹{height:,.0f}', ha='center', va='bottom', rotation=0)
            
            fig.tight_layout()
            st.pyplot(fig)
        else:
            st.info("No sales data available for the selected time period.")
        
        # Category Distribution
        st.subheader("Sales by Category")
        category_sales = execute_query(f"""
            SELECT d.category, SUM(s.total_price) as category_sales
            FROM sales s JOIN drugs d ON s.drug_id = d.id
            WHERE 1=1 {date_filter}
            GROUP BY d.category
            ORDER BY category_sales DESC
        """)
        
        if not category_sales.empty:
            # Create two columns for charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Pie chart for category distribution
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.pie(category_sales['category_sales'], labels=category_sales['category'], 
                      autopct='%1.1f%%', startangle=90, shadow=True)
                ax.axis('equal')
                ax.set_title('Sales Distribution by Drug Category')
                fig.tight_layout()
                st.pyplot(fig)
            
            with col2:
                # Bar chart for category sales
                fig, ax = plt.subplots(figsize=(10, 6))
                category_sales = category_sales.sort_values('category_sales')
                bars = ax.barh(category_sales['category'], category_sales['category_sales'])
                ax.set_xlabel('Sales (₹)')
                ax.set_title('Sales by Category')
                
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width + width*0.01, bar.get_y() + bar.get_height()/2,
                            f'₹{width:,.0f}', va='center')
                
                fig.tight_layout()
                st.pyplot(fig)
        else:
            st.info("No sales data available for the selected time period.")
            
        # Recent Sales Table (limited to 10 rows)
        st.subheader("Recent Sales")
        recent_sales = execute_query(f"""
            SELECT 
                date(s.sale_date) as Date,
                d.name as Product,
                d.category as Category,
                s.quantity as Quantity,
                s.total_price as Revenue
            FROM sales s 
            JOIN drugs d ON s.drug_id = d.id
            WHERE 1=1 {date_filter}
            ORDER BY s.sale_date DESC
            LIMIT 10
        """)
        
        if not recent_sales.empty:
            st.dataframe(recent_sales, use_container_width=True)
        else:
            st.info("No recent sales data available.")
    
    except Exception as e:
        logger.error(f"Error in dashboard: {e}")
        st.error(f"An error occurred while loading the dashboard: {str(e)}")