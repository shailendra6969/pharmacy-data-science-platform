"""
SQLite database handling for the Pharmacy Data Science Platform.
"""
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import random
import os
import time
from config import DB_PATH, logger

def get_db_connection():
    """Create and return a database connection with error handling and retry logic"""
    max_retries = 3
    retry_delay = 1  # seconds
    
    for attempt in range(max_retries):
        try:
            conn = sqlite3.connect(DB_PATH, timeout=20)
            # Enable foreign keys
            conn.execute("PRAGMA foreign_keys = ON")
            # Return successful connection
            return conn
        except sqlite3.Error as e:
            logger.error(f"Database connection error (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                raise Exception(f"Failed to connect to database after {max_retries} attempts: {e}")

def check_and_initialize_db(drugs_df):
    """Check if database exists and initialize it if needed"""
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        
        # Create connection
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Create drugs table with expanded schema
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS drugs (
            id INTEGER PRIMARY KEY,
            generic_name TEXT NOT NULL,
            brand_name TEXT NOT NULL,
            manufacturer TEXT,
            category TEXT,
            subcategory TEXT,
            ndc TEXT,
            price REAL,
            dosage_form TEXT,
            dosage TEXT,
            description TEXT,
            gene_interactions TEXT,
            stock INTEGER
        )
        ''')
        
        # Create sales table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS sales (
            id INTEGER PRIMARY KEY,
            drug_id INTEGER,
            sale_date TEXT,
            quantity INTEGER,
            total_price REAL,
            FOREIGN KEY (drug_id) REFERENCES drugs (id)
        )
        ''')
        
        # Check if drugs table is empty
        cursor.execute("SELECT COUNT(*) FROM drugs")
        if cursor.fetchone()[0] == 0 and drugs_df is not None:
            # Check if dataframe has the required columns
            required_columns = ['id', 'generic_name', 'brand_name', 'manufacturer', 'category',
                              'subcategory', 'price', 'dosage_form', 'dosage', 'description',
                              'stock']
            
            missing_columns = [col for col in required_columns if col not in drugs_df.columns]
            
            if missing_columns:
                logger.warning(f"Missing columns in drugs dataframe: {missing_columns}")
                # Add missing columns with default values
                for col in missing_columns:
                    if col == 'id':
                        drugs_df[col] = range(1, len(drugs_df) + 1)
                    elif col in ['price', 'stock']:
                        drugs_df[col] = 0
                    else:
                        drugs_df[col] = ""
            
            # Truncate long description fields if necessary
            if 'description' in drugs_df.columns:
                drugs_df['description'] = drugs_df['description'].str.slice(0, 10000)
            
            # Define the correct column order based on table schema
            table_columns = ['id', 'generic_name', 'brand_name', 'manufacturer', 'category',
                           'subcategory', 'ndc', 'price', 'dosage_form', 'dosage',
                           'description', 'gene_interactions', 'stock']
            
            # Add any missing columns to dataframe with default values
            for col in table_columns:
                if col not in drugs_df.columns:
                    if col in ['price', 'stock']:
                        drugs_df[col] = 0
                    else:
                        drugs_df[col] = ""
            
            # Select and reorder columns according to table schema
            drugs_df_to_insert = drugs_df[table_columns]
            
            # Insert data
            try:
                # Convert to list of tuples for better insertion performance
                drugs_data = [tuple(row) for row in drugs_df_to_insert.itertuples(index=False)]
                
                # Use UPSERT with INSERT OR REPLACE to handle potential duplicates
                insert_sql = f"INSERT OR REPLACE INTO drugs VALUES ({','.join(['?'] * len(table_columns))})"
                
                # Insert in batches to avoid SQLite limits
                batch_size = 100
                for i in range(0, len(drugs_data), batch_size):
                    batch = drugs_data[i:i+batch_size]
                    cursor.executemany(insert_sql, batch)
                    conn.commit()
                
                logger.info(f"Loaded {len(drugs_data)} drugs into database")
            except sqlite3.Error as e:
                logger.error(f"Error inserting drug data: {e}")
                conn.rollback()
                raise
            
            # Generate sample sales data
            generate_sample_sales_data(conn, cursor)
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise

def generate_sample_sales_data(conn, cursor):
    """Generate realistic sample sales data for all drugs"""
    try:
        # Get drug IDs and prices
        cursor.execute("SELECT id, category, price FROM drugs")
        drugs = cursor.fetchall()
        
        if not drugs:
            logger.warning("No drugs found in database, skipping sales data generation")
            return
        
        sample_sales = []
        sale_id = 1
        
        # Generate 365 days of sales data
        for day in range(365):
            date = (datetime.now() - timedelta(days=365-day)).strftime('%Y-%m-%d')
            date_obj = datetime.strptime(date, '%Y-%m-%d')
            is_weekend = (date_obj.weekday() >= 5)
            month = date_obj.month
            is_winter = (month in [12, 1, 2])
            is_summer = (month in [6, 7, 8])
            
            for drug_id, category, price in drugs:
                # Different sales patterns based on day of week, season, and category
                if is_weekend:
                    base_quantity = random.randint(3, 8)
                else:
                    base_quantity = random.randint(1, 5)
                
                # Seasonal adjustments
                if category in ["Respiratory", "Antibiotic"] and is_winter:
                    quantity = base_quantity * 2
                elif category in ["Allergy", "Dermatological"] and is_summer:
                    quantity = base_quantity * 1.5
                else:
                    quantity = base_quantity
                
                # Random fluctuations (±20%)
                quantity = max(1, int(quantity * random.uniform(0.8, 1.2)))
                
                total = price * quantity
                sample_sales.append((sale_id, drug_id, date, quantity, total))
                sale_id += 1
                
            # Commit in batches to avoid memory issues
            if day % 30 == 0 and sample_sales:
                try:
                    cursor.executemany("INSERT INTO sales VALUES (?, ?, ?, ?, ?)", sample_sales)
                    conn.commit()
                    sample_sales = []
                    logger.info(f"Generated sales data for {day} days")
                except sqlite3.Error as e:
                    logger.error(f"Error inserting batch of sales data: {e}")
                    conn.rollback()
        
        # Insert any remaining records
        if sample_sales:
            try:
                cursor.executemany("INSERT INTO sales VALUES (?, ?, ?, ?, ?)", sample_sales)
                conn.commit()
            except sqlite3.Error as e:
                logger.error(f"Error inserting final batch of sales data: {e}")
                conn.rollback()
        
        logger.info("Sample sales data generation complete")
        
    except Exception as e:
        logger.error(f"Error generating sample sales data: {e}")
        raise

def execute_query(query, params=None, fetch=True):
    """Execute a SQL query with error handling and retry logic"""
    conn = None
    max_retries = 3
    retry_delay = 1  # seconds
    
    for attempt in range(max_retries):
        try:
            conn = get_db_connection()
            
            if fetch:
                result = pd.read_sql(query, conn, params=params)
                conn.close()
                return result
            else:
                cursor = conn.cursor()
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                conn.commit()
                last_id = cursor.lastrowid
                conn.close()
                return last_id
                
        except (sqlite3.Error, pd.io.sql.DatabaseError) as e:
            if conn:
                conn.close()
            
            error_msg = str(e)
            logger.error(f"Query execution error (attempt {attempt+1}/{max_retries}): {error_msg}")
            
            # Check for specific error types
            if "no such column" in error_msg.lower():
                raise Exception(f"Column not found in database: {error_msg}")
            elif "no such table" in error_msg.lower():
                raise Exception(f"Table not found in database: {error_msg}")
            
            # Retry for other errors
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                raise Exception(f"Failed to execute query after {max_retries} attempts: {error_msg}")
    
    # This should not be reached, but just in case
    raise Exception("Unexpected error in execute_query")