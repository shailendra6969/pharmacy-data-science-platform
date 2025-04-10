"""
Product Catalog module for the Pharmacy Data Science Platform.
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import logger
from db.sqlite_db import execute_query

def show_product_catalog():
    """Display the Product Catalog module"""
    st.subheader("📋 Pharmaceutical Product Catalog")
    st.markdown("""
    This module provides a comprehensive view of the pharmaceutical products in the inventory.
    You can search, filter, and analyze products by various attributes.
    """)
    
    try:
        # Load all drug data from database
        drugs_df = execute_query("""
            SELECT 
                d.id, 
                d.name, 
                d.category, 
                d.price, 
                d.dosage, 
                d.description, 
                d.manufacturer, 
                d.stock,
                SUM(s.quantity) as total_sold,
                SUM(s.total_price) as total_revenue
            FROM drugs d
            LEFT JOIN sales s ON d.id = s.drug_id
            GROUP BY d.id
            ORDER BY d.name
        """)
        
        if drugs_df.empty:
            st.warning("No product data available in the database.")
            return
        
        # Create search and filter options
        st.subheader("Search and Filter Products")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Text search
            search_term = st.text_input("Search by Name or Description:", "")
        
        with col2:
            # Category filter
            categories = ["All Categories"] + sorted(drugs_df['category'].unique().tolist())
            selected_category = st.selectbox("Filter by Category:", categories)
        
        with col3:
            # Manufacturer filter
            manufacturers = ["All Manufacturers"] + sorted(drugs_df['manufacturer'].unique().tolist())
            selected_manufacturer = st.selectbox("Filter by Manufacturer:", manufacturers)
        
        # Apply filters
        filtered_df = drugs_df.copy()
        
        if search_term:
            filtered_df = filtered_df[
                filtered_df['name'].str.contains(search_term, case=False) | 
                filtered_df['description'].str.contains(search_term, case=False)
            ]
        
        if selected_category != "All Categories":
            filtered_df = filtered_df[filtered_df['category'] == selected_category]
        
        if selected_manufacturer != "All Manufacturers":
            filtered_df = filtered_df[filtered_df['manufacturer'] == selected_manufacturer]
        
        # Additional filters
        col1, col2 = st.columns(2)
        
        with col1:
            # Price range filter
            min_price = float(drugs_df['price'].min())
            max_price = float(drugs_df['price'].max())
            price_range = st.slider(
                "Price Range (₹):", 
                min_value=min_price,
                max_value=max_price,
                value=(min_price, max_price),
                step=10.0
            )
            filtered_df = filtered_df[
                (filtered_df['price'] >= price_range[0]) & 
                (filtered_df['price'] <= price_range[1])
            ]
        
        with col2:
            # Stock status filter
            stock_status = st.radio(
                "Stock Status:",
                ["All", "In Stock", "Low Stock", "Out of Stock"],
                horizontal=True
            )
            
            if stock_status == "In Stock":
                filtered_df = filtered_df[filtered_df['stock'] > 400]
            elif stock_status == "Low Stock":
                filtered_df = filtered_df[(filtered_df['stock'] > 0) & (filtered_df['stock'] <= 400)]
            elif stock_status == "Out of Stock":
                filtered_df = filtered_df[filtered_df['stock'] <= 0]
        
        # Sort options
        sort_by = st.selectbox(
            "Sort By:",
            ["Name (A-Z)", "Name (Z-A)", "Price (Low to High)", "Price (High to Low)", 
             "Sales Volume (High to Low)", "Revenue (High to Low)"]
        )
        
        if sort_by == "Name (A-Z)":
            filtered_df = filtered_df.sort_values('name')
        elif sort_by == "Name (Z-A)":
            filtered_df = filtered_df.sort_values('name', ascending=False)
        elif sort_by == "Price (Low to High)":
            filtered_df = filtered_df.sort_values('price')
        elif sort_by == "Price (High to Low)":
            filtered_df = filtered_df.sort_values('price', ascending=False)
        elif sort_by == "Sales Volume (High to Low)":
            filtered_df = filtered_df.sort_values('total_sold', ascending=False)
        elif sort_by == "Revenue (High to Low)":
            filtered_df = filtered_df.sort_values('total_revenue', ascending=False)
        
        # Display results
        st.subheader(f"Results: {len(filtered_df)} Products Found")
        
        if filtered_df.empty:
            st.info("No products match your search criteria. Try adjusting the filters.")
        else:
            # Format DataFrame for display
            display_df = filtered_df[['name', 'category', 'price', 'dosage', 'manufacturer', 'stock', 'total_sold']].copy()
            display_df['price'] = display_df['price'].apply(lambda x: f"₹{x:,.2f}")
            
            # Show results as a table
            st.dataframe(display_df, use_container_width=True)
            
            # Product Detail View
            st.subheader("Product Details")
            product_names = filtered_df['name'].tolist()
            selected_product = st.selectbox("Select a product to view details:", product_names)
            
            product_details = filtered_df[filtered_df['name'] == selected_product].iloc[0]
            
            # Display product details
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                st.markdown(f"**ID:** {product_details['id']}")
                st.markdown(f"**Name:** {product_details['name']}")
                st.markdown(f"**Category:** {product_details['category']}")
                st.markdown(f"**Price:** ₹{product_details['price']:,.2f}")
            
            with col2:
                st.markdown(f"**Manufacturer:** {product_details['manufacturer']}")
                st.markdown(f"**Dosage:** {product_details['dosage']}")
                st.markdown(f"**Current Stock:** {product_details['stock']} units")
                
                # Color-code stock status
                if product_details['stock'] <= 0:
                    st.error("Out of Stock")
                elif product_details['stock'] <= 400:
                    st.warning("Low Stock")
                else:
                    st.success("In Stock")
            
            with col3:
                st.markdown(f"**Total Units Sold:** {product_details['total_sold'] if pd.notna(product_details['total_sold']) else 0:,.0f}")
                st.markdown(f"**Total Revenue:** ₹{product_details['total_revenue'] if pd.notna(product_details['total_revenue']) else 0:,.2f}")
                
                # Calculate profit margin (dummy calculation for demonstration)
                cost_price = product_details['price'] * 0.6  # Assuming cost is 60% of selling price
                margin = product_details['price'] - cost_price
                margin_percent = (margin / product_details['price']) * 100
                st.markdown(f"**Estimated Margin:** {margin_percent:.1f}%")
            
            # Product description
            st.subheader("Description")
            st.markdown(product_details['description'])
            
            # Sales history for the selected product
            try:
                sales_history = execute_query("""
                    SELECT date(sale_date) as date, SUM(quantity) as units_sold, SUM(total_price) as revenue
                    FROM sales
                    WHERE drug_id = ?
                    GROUP BY date
                    ORDER BY date
                """, params=(product_details['id'],))
                
                if not sales_history.empty and len(sales_history) > 1:
                    st.subheader("Sales History")
                    
                    # Line chart for sales over time
                    sales_history['date'] = pd.to_datetime(sales_history['date'])
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(sales_history['date'], sales_history['units_sold'], 'b-', marker='o', markersize=4)
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Units Sold')
                    ax.set_title(f'Sales History: {selected_product}')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
            except Exception as e:
                logger.error(f"Error fetching sales history: {e}")
                st.error(f"Error fetching sales history: {str(e)}")
        
        # Analytics Section
        st.subheader("Catalog Analytics")
        tab1, tab2 = st.tabs(["Price Distribution", "Stock Analysis"])
        
        with tab1:
            # Price distribution by category
            st.subheader("Price Distribution by Category")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(x='category', y='price', data=drugs_df, ax=ax)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Price histogram
            st.subheader("Price Distribution Histogram")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(drugs_df['price'], bins=20, kde=True, ax=ax)
            ax.set_xlabel('Price (₹)')
            ax.set_ylabel('Number of Products')
            plt.tight_layout()
            st.pyplot(fig)
        
        with tab2:
            # Stock level analysis
            st.subheader("Stock Level Analysis")
            
            # Define stock status
            drugs_df['stock_status'] = pd.cut(
                drugs_df['stock'],
                bins=[-1, 0, 400, float('inf')],
                labels=['Out of Stock', 'Low Stock', 'In Stock']
            )
            
            # Calculate percentages
            stock_count = drugs_df['stock_status'].value_counts()
            stock_percent = drugs_df['stock_status'].value_counts(normalize=True) * 100
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Pie chart of stock status
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.pie(
                    stock_count,
                    labels=stock_count.index,
                    autopct='%1.1f%%',
                    startangle=90,
                    colors=['#ff9999', '#ffcc99', '#99ff99']
                )
                ax.axis('equal')
                ax.set_title('Stock Status Distribution')
                st.pyplot(fig)
            
            with col2:
                # Stock by category
                stock_by_category = drugs_df.groupby('category')['stock'].sum().sort_values(ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                stock_by_category.plot(kind='bar', ax=ax)
                ax.set_xlabel('Category')
                ax.set_ylabel('Total Stock')
                ax.set_title('Total Stock by Category')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
        
        # Export option
        if st.button("Export Filtered Products to CSV"):
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="product_catalog.csv",
                mime="text/csv"
            )
    
    except Exception as e:
        logger.error(f"Error in product catalog: {e}")
        st.error(f"Error loading product catalog: {str(e)}")