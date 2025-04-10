"""
SQLite Database Explorer module for the Pharmacy Data Science Platform.
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import re
from config import logger, DB_PATH
from db.sqlite_db import execute_query

def show_database_explorer():
    """Display the SQLite Database Explorer module"""
    st.subheader("🗄️ SQLite Database Explorer")
    st.markdown("""
    This module allows you to explore and query the SQLite database directly.
    View table schemas, run custom SQL queries, and visualize the results.
    """)
    
    try:
        # Connect to database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get list of tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [table[0] for table in cursor.fetchall()]
        
        if not tables:
            st.warning("No tables found in the database.")
            return
        
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["Database Schema", "SQL Query", "Data Visualization"])
        
        # ----- Database Schema Tab -----
        with tab1:
            st.subheader("Database Schema")
            
            # Create a dictionary to store table information
            table_info = {}
            
            # Get schema for each table
            for table in tables:
                cursor.execute(f"PRAGMA table_info({table})")
                columns = cursor.fetchall()
                
                # Format column information
                column_info = []
                for col in columns:
                    column_info.append({
                        "cid": col[0],
                        "name": col[1],
                        "type": col[2],
                        "notnull": "NOT NULL" if col[3] else "",
                        "default": col[4],
                        "pk": "PRIMARY KEY" if col[5] else ""
                    })
                
                # Add to table info dictionary
                table_info[table] = column_info
                
                # Get row count
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                row_count = cursor.fetchone()[0]
                
                # Display table schema
                st.markdown(f"### Table: `{table}` ({row_count} rows)")
                
                # Create DataFrame for display
                schema_df = pd.DataFrame(column_info)
                schema_df = schema_df[['name', 'type', 'notnull', 'pk', 'default']]
                schema_df.columns = ["Column Name", "Data Type", "Not Null", "Primary Key", "Default Value"]
                
                st.dataframe(schema_df, use_container_width=True)
                
                # Get foreign key information
                cursor.execute(f"PRAGMA foreign_key_list({table})")
                foreign_keys = cursor.fetchall()
                
                if foreign_keys:
                    st.markdown("**Foreign Keys:**")
                    fk_data = []
                    for fk in foreign_keys:
                        fk_data.append({
                            "Column": fk[3],
                            "References": f"{fk[2]}({fk[4]})",
                            "On Update": fk[5],
                            "On Delete": fk[6]
                        })
                    
                    st.dataframe(pd.DataFrame(fk_data), use_container_width=True)
                
                # Preview of data
                cursor.execute(f"SELECT * FROM {table} LIMIT 5")
                preview_data = cursor.fetchall()
                
                if preview_data:
                    st.markdown("**Data Preview:**")
                    column_names = [description[0] for description in cursor.description]
                    preview_df = pd.DataFrame(preview_data, columns=column_names)
                    st.dataframe(preview_df, use_container_width=True)
                
                st.markdown("---")
            
            # Display entity relationship information
            st.subheader("Entity Relationships")
            st.markdown("""
            The database contains the following key relationships:
            
            - **drugs**: Primary table containing drug information
            - **sales**: Sales records linked to drugs via `drug_id` foreign key
            """)
        
        # ----- SQL Query Tab -----
        with tab2:
            st.subheader("SQL Query Tool")
            
            # Sample queries dropdown
            sample_queries = {
                "Select a sample query...": "",
                "List all drugs": "SELECT * FROM drugs",
                "Top 10 bestselling drugs": """
                    SELECT d.name, d.category, SUM(s.quantity) as total_sold, SUM(s.total_price) as total_revenue 
                    FROM drugs d 
                    JOIN sales s ON d.id = s.drug_id 
                    GROUP BY d.id 
                    ORDER BY total_sold DESC 
                    LIMIT 10
                """,
                "Monthly sales by category": """
                    SELECT 
                        strftime('%Y-%m', s.sale_date) as month, 
                        d.category, 
                        SUM(s.total_price) as monthly_revenue 
                    FROM sales s 
                    JOIN drugs d ON s.drug_id = d.id 
                    GROUP BY month, d.category 
                    ORDER BY month, monthly_revenue DESC
                """,
                "Low stock drugs": """
                    SELECT * FROM drugs WHERE stock < 400 ORDER BY stock
                """,
                "Sales by day of week": """
                    SELECT 
                        strftime('%w', s.sale_date) as day_of_week, 
                        CASE 
                            WHEN strftime('%w', s.sale_date) = '0' THEN 'Sunday'
                            WHEN strftime('%w', s.sale_date) = '1' THEN 'Monday'
                            WHEN strftime('%w', s.sale_date) = '2' THEN 'Tuesday'
                            WHEN strftime('%w', s.sale_date) = '3' THEN 'Wednesday'
                            WHEN strftime('%w', s.sale_date) = '4' THEN 'Thursday'
                            WHEN strftime('%w', s.sale_date) = '5' THEN 'Friday'
                            WHEN strftime('%w', s.sale_date) = '6' THEN 'Saturday'
                        END as day_name,
                        SUM(s.quantity) as total_units,
                        SUM(s.total_price) as total_sales
                    FROM sales s
                    GROUP BY day_of_week
                    ORDER BY day_of_week
                """
            }
            
            selected_sample = st.selectbox("Sample Queries:", list(sample_queries.keys()))
            
            # Set default query text
            default_query = sample_queries[selected_sample].strip() if selected_sample in sample_queries else ""
            
            # SQL query input
            sql_query = st.text_area("Enter SQL Query:", default_query, height=150)
            
            # Query execution controls
            col1, col2 = st.columns([1, 4])
            with col1:
                execute_button = st.button("Execute Query")
            with col2:
                st.markdown("")  # Empty space for alignment
            
            # Execute query when button is clicked
            if execute_button and sql_query:
                # Check if the query is safe (basic check to prevent data modification)
                query_lower = sql_query.lower().strip()
                
                # List of potentially dangerous operations
                dangerous_operations = ['drop', 'delete', 'truncate', 'update', 'insert', 'alter', 'create']
                
                # Check if query starts with any dangerous operation
                is_dangerous = any(query_lower.startswith(op) for op in dangerous_operations)
                
                if is_dangerous:
                    st.error("""
                    For safety reasons, this interface only allows SELECT queries. 
                    Data modification operations are disabled in this view.
                    """)
                else:
                    try:
                        # Execute the query
                        start_time = pd.Timestamp.now()
                        result_df = pd.read_sql(sql_query, conn)
                        end_time = pd.Timestamp.now()
                        execution_time = (end_time - start_time).total_seconds()
                        
                        # Display query results
                        st.success(f"Query executed successfully in {execution_time:.3f} seconds. {len(result_df)} rows returned.")
                        
                        if not result_df.empty:
                            st.subheader("Query Results")
                            st.dataframe(result_df, use_container_width=True)
                            
                            # Option to download results
                            csv = result_df.to_csv(index=False)
                            st.download_button(
                                label="Download Results as CSV",
                                data=csv,
                                file_name="query_results.csv",
                                mime="text/csv"
                            )
                            
                            # Save result to session state for visualization
                            st.session_state.query_result = result_df
                        else:
                            st.info("The query returned no results.")
                    
                    except sqlite3.Error as e:
                        logger.error(f"SQL error: {e}")
                        st.error(f"SQL Error: {str(e)}")
                    
                    except Exception as e:
                        logger.error(f"Error executing query: {e}")
                        st.error(f"Error: {str(e)}")
        
        # ----- Data Visualization Tab -----
        with tab3:
            st.subheader("Data Visualization")
            
            if 'query_result' not in st.session_state:
                st.info("Run a query in the 'SQL Query' tab to visualize the results.")
            else:
                result_df = st.session_state.query_result
                
                if result_df.empty:
                    st.warning("No data to visualize.")
                else:
                    # Identify numeric and categorical columns
                    numeric_cols = result_df.select_dtypes(include=['number']).columns.tolist()
                    categorical_cols = result_df.select_dtypes(include=['object', 'category']).columns.tolist()
                    date_cols = []
                    
                    # Try to identify date columns
                    for col in result_df.columns:
                        if col in categorical_cols:
                            # Check if column contains dates
                            if any(re.match(r'^\d{4}-\d{2}-\d{2}', str(val)) for val in result_df[col].dropna().head(10)):
                                date_cols.append(col)
                                categorical_cols.remove(col)
                    
                    # Chart type selection
                    chart_type = st.selectbox(
                        "Select Chart Type:",
                        ["Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart", "Histogram", "Box Plot", "Heatmap"]
                    )
                    
                    # Chart configuration
                    st.subheader("Chart Configuration")
                    
                    # Bar Chart
                    if chart_type == "Bar Chart":
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            x_axis = st.selectbox("X-Axis (Categories):", categorical_cols if categorical_cols else result_df.columns.tolist())
                        
                        with col2:
                            y_axis = st.selectbox("Y-Axis (Values):", numeric_cols if numeric_cols else result_df.columns.tolist())
                        
                        # Additional options
                        use_horizontal = st.checkbox("Horizontal Bar Chart")
                        
                        # Create the chart
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        if use_horizontal:
                            result_df.sort_values(y_axis, ascending=True).plot(
                                kind='barh', x=x_axis, y=y_axis, ax=ax
                            )
                            ax.set_xlabel(y_axis)
                            ax.set_ylabel(x_axis)
                        else:
                            result_df.plot(kind='bar', x=x_axis, y=y_axis, ax=ax)
                            ax.set_xlabel(x_axis)
                            ax.set_ylabel(y_axis)
                        
                        ax.set_title(f"{y_axis} by {x_axis}")
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    # Line Chart
                    elif chart_type == "Line Chart":
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            x_axis = st.selectbox(
                                "X-Axis:", 
                                date_cols + categorical_cols + numeric_cols
                            )
                        
                        with col2:
                            y_axis = st.selectbox(
                                "Y-Axis:", 
                                numeric_cols if numeric_cols else result_df.columns.tolist()
                            )
                        
                        with col3:
                            group_by = st.selectbox(
                                "Group By (optional):", 
                                ["None"] + categorical_cols
                            )
                        
                        # Create the chart
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        if group_by != "None":
                            for group, data in result_df.groupby(group_by):
                                data.plot(kind='line', x=x_axis, y=y_axis, ax=ax, label=group)
                            plt.legend(title=group_by)
                        else:
                            result_df.plot(kind='line', x=x_axis, y=y_axis, ax=ax)
                        
                        ax.set_xlabel(x_axis)
                        ax.set_ylabel(y_axis)
                        ax.set_title(f"{y_axis} over {x_axis}")
                        
                        # Rotate x labels if there are many categories
                        if len(result_df[x_axis].unique()) > 8:
                            plt.xticks(rotation=45, ha='right')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    # Scatter Plot
                    elif chart_type == "Scatter Plot":
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            x_axis = st.selectbox(
                                "X-Axis:", 
                                numeric_cols if numeric_cols else result_df.columns.tolist()
                            )
                        
                        with col2:
                            y_axis = st.selectbox(
                                "Y-Axis:", 
                                [col for col in numeric_cols if col != x_axis] if len(numeric_cols) > 1 else numeric_cols
                            )
                        
                        with col3:
                            color_by = st.selectbox(
                                "Color By (optional):", 
                                ["None"] + categorical_cols
                            )
                        
                        # Create the chart
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        if color_by != "None":
                            for group, data in result_df.groupby(color_by):
                                ax.scatter(data[x_axis], data[y_axis], label=group, alpha=0.7)
                            plt.legend(title=color_by)
                        else:
                            ax.scatter(result_df[x_axis], result_df[y_axis], alpha=0.7)
                        
                        ax.set_xlabel(x_axis)
                        ax.set_ylabel(y_axis)
                        ax.set_title(f"{y_axis} vs {x_axis}")
                        plt.grid(alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    # Pie Chart
                    elif chart_type == "Pie Chart":
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            labels = st.selectbox("Categories:", categorical_cols if categorical_cols else result_df.columns.tolist())
                        
                        with col2:
                            values = st.selectbox("Values:", numeric_cols if numeric_cols else result_df.columns.tolist())
                        
                        # For pie charts, we need to aggregate the data
                        pie_data = result_df.groupby(labels)[values].sum()
                        
                        # Create the chart
                        fig, ax = plt.subplots(figsize=(10, 6))
                        pie_data.plot(kind='pie', ax=ax, autopct='%1.1f%%')
                        ax.set_ylabel('')  # Remove the automatic ylabel
                        ax.set_title(f"Distribution of {values} by {labels}")
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    # Histogram
                    elif chart_type == "Histogram":
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            column = st.selectbox(
                                "Select Column:", 
                                numeric_cols if numeric_cols else result_df.columns.tolist()
                            )
                        
                        with col2:
                            bins = st.slider("Number of Bins:", 5, 100, 20)
                        
                        # Create the chart
                        fig, ax = plt.subplots(figsize=(10, 6))
                        result_df[column].plot(kind='hist', bins=bins, ax=ax)
                        ax.set_xlabel(column)
                        ax.set_ylabel('Frequency')
                        ax.set_title(f"Histogram of {column}")
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    # Box Plot
                    elif chart_type == "Box Plot":
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            value_col = st.selectbox(
                                "Values:", 
                                numeric_cols if numeric_cols else result_df.columns.tolist()
                            )
                        
                        with col2:
                            group_col = st.selectbox(
                                "Group By (optional):", 
                                ["None"] + categorical_cols
                            )
                        
                        # Create the chart
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        if group_col != "None":
                            sns.boxplot(x=group_col, y=value_col, data=result_df, ax=ax)
                            ax.set_xlabel(group_col)
                        else:
                            sns.boxplot(y=value_col, data=result_df, ax=ax)
                            ax.set_xlabel('')
                        
                        ax.set_ylabel(value_col)
                        ax.set_title(f"Box Plot of {value_col}")
                        
                        # Rotate x labels if there are many categories
                        if group_col != "None" and len(result_df[group_col].unique()) > 4:
                            plt.xticks(rotation=45, ha='right')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    # Heatmap
                    elif chart_type == "Heatmap":
                        if len(numeric_cols) < 2:
                            st.warning("Heatmap requires at least 2 numeric columns.")
                        else:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                index_col = st.selectbox(
                                    "Rows:", 
                                    categorical_cols if categorical_cols else result_df.columns.tolist()
                                )
                            
                            with col2:
                                columns_col = st.selectbox(
                                    "Columns:", 
                                    [col for col in categorical_cols if col != index_col] if len(categorical_cols) > 1 else categorical_cols
                                )
                            
                            value_col = st.selectbox(
                                "Values:", 
                                numeric_cols if numeric_cols else result_df.columns.tolist()
                            )
                            
                            # Create pivot table for heatmap
                            try:
                                pivot_table = pd.pivot_table(
                                    result_df, 
                                    values=value_col, 
                                    index=index_col,
                                    columns=columns_col,
                                    aggfunc='mean'
                                )
                                
                                # Create the chart
                                fig, ax = plt.subplots(figsize=(12, 8))
                                sns.heatmap(pivot_table, annot=True, cmap='YlGnBu', ax=ax)
                                ax.set_title(f"Heatmap of {value_col} by {index_col} and {columns_col}")
                                plt.tight_layout()
                                st.pyplot(fig)
                            
                            except Exception as e:
                                logger.error(f"Error creating heatmap: {e}")
                                st.error(f"Error creating heatmap: {str(e)}")
                                st.info("Heatmap may not work with this data structure. Try a different visualization or adjust the query.")
                    
                    # Option to save the chart
                    st.markdown("Right-click on the chart to save it as an image.")
        
        # Close database connection
        conn.close()
    
    except Exception as e:
        logger.error(f"Error in database explorer: {e}")
        st.error(f"Error: {str(e)}")