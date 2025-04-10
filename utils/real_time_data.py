"""
Real-Time Data Integration for the Pharmacy Data Science Platform.

This module provides functionality for real-time data updates and integration
with external data sources for live monitoring of pharmacy metrics.
"""
import streamlit as st
import pandas as pd
import numpy as np
import threading
import time
import json
import os
import requests
from datetime import datetime, timedelta
import sqlite3
import threading
import queue
from config import logger, DB_PATH, DATA_DIR

# Define update intervals (in seconds)
UPDATE_INTERVALS = {
    "price_data": 3600,  # 1 hour
    "stock_data": 900,   # 15 minutes
    "sales_data": 300,   # 5 minutes
    "market_data": 3600, # 1 hour
}

# Last update timestamps
last_updates = {key: datetime.min for key in UPDATE_INTERVALS.keys()}

# Global data storage for real-time data
realtime_data = {
    "price_data": None,
    "stock_data": None,
    "sales_data": None,
    "market_data": None,
}

# Data sources configuration (would be loaded from config in a real application)
DATA_SOURCES = {
    "price_updates": None,  # Update this with actual API URL in production
    "stock_updates": None,  # Update this with actual API URL in production
    "sales_updates": None,  # Update this with actual API URL in production
    "market_data": None,    # Update this with actual API URL in production
}

# Task queue for background updates
update_queue = queue.Queue()

# Flag to track if the background worker is running
is_worker_running = False

def initialize_realtime_data():
    """Initialize real-time data from local database"""
    try:
        # Set initial data from database
        conn = sqlite3.connect(DB_PATH)
        
        # Get current drug prices
        price_data = pd.read_sql("""
            SELECT id, generic_name, brand_name, price
            FROM drugs
        """, conn)
        realtime_data["price_data"] = price_data
        
        # Get current stock levels
        stock_data = pd.read_sql("""
            SELECT id, generic_name, brand_name, stock
            FROM drugs
        """, conn)
        realtime_data["stock_data"] = stock_data
        
        # Get recent sales data
        sales_data = pd.read_sql("""
            SELECT date(sale_date) as date, SUM(total_price) as daily_sales
            FROM sales
            WHERE date(sale_date) >= date('now', '-30 days')
            GROUP BY date
            ORDER BY date
        """, conn)
        realtime_data["sales_data"] = sales_data
        
        # Simulated market data
        market_data = pd.DataFrame({
            "date": pd.date_range(end=datetime.now(), periods=30, freq='D'),
            "market_index": np.cumsum(np.random.normal(0, 1, 30)) + 100,
            "volume": np.random.randint(10000, 50000, 30)
        })
        realtime_data["market_data"] = market_data
        
        conn.close()
        
        # Update timestamps
        current_time = datetime.now()
        for key in last_updates:
            last_updates[key] = current_time
        
        logger.info("Real-time data initialized successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error initializing real-time data: {e}")
        return False

def data_needs_update(data_type):
    """Check if a data type needs to be updated based on interval"""
    if data_type not in UPDATE_INTERVALS:
        return False
    
    current_time = datetime.now()
    interval = UPDATE_INTERVALS[data_type]
    last_update = last_updates.get(data_type, datetime.min)
    
    return (current_time - last_update).total_seconds() >= interval

def update_price_data():
    """Update price data from external source or simulate changes"""
    try:
        # If real API configured, use it
        if DATA_SOURCES["price_updates"]:
            response = requests.get(DATA_SOURCES["price_updates"], timeout=10)
            if response.status_code == 200:
                new_price_data = pd.DataFrame(response.json())
                if not new_price_data.empty:
                    realtime_data["price_data"] = new_price_data
                    last_updates["price_data"] = datetime.now()
                    return True
        
        # Otherwise, simulate price changes
        if realtime_data["price_data"] is not None:
            price_df = realtime_data["price_data"].copy()
            
            # Update random 5% of drugs with small price changes
            indices = np.random.choice(
                price_df.index, 
                size=max(1, int(len(price_df) * 0.05)), 
                replace=False
            )
            
            for idx in indices:
                current_price = price_df.loc[idx, 'price']
                # Random change between -2% and +2%
                change_pct = np.random.uniform(-0.02, 0.02)
                new_price = max(0.1, current_price * (1 + change_pct))
                price_df.loc[idx, 'price'] = round(new_price, 2)
            
            # Update cache
            realtime_data["price_data"] = price_df
            
            # Update database with new prices
            try:
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                
                for idx, row in price_df.iterrows():
                    cursor.execute(
                        "UPDATE drugs SET price = ? WHERE id = ?",
                        (row['price'], row['id'])
                    )
                
                conn.commit()
                conn.close()
            except Exception as e:
                logger.error(f"Error updating database with new prices: {e}")
            
            last_updates["price_data"] = datetime.now()
            logger.info(f"Simulated price updates for {len(indices)} drugs")
            return True
    
    except Exception as e:
        logger.error(f"Error updating price data: {e}")
    
    return False

def update_stock_data():
    """Update stock data from external source or simulate changes"""
    try:
        # If real API configured, use it
        if DATA_SOURCES["stock_updates"]:
            response = requests.get(DATA_SOURCES["stock_updates"], timeout=10)
            if response.status_code == 200:
                new_stock_data = pd.DataFrame(response.json())
                if not new_stock_data.empty:
                    realtime_data["stock_data"] = new_stock_data
                    last_updates["stock_data"] = datetime.now()
                    return True
        
        # Otherwise, simulate stock changes
        if realtime_data["stock_data"] is not None:
            stock_df = realtime_data["stock_data"].copy()
            
            # Update random 10% of drugs with stock changes
            indices = np.random.choice(
                stock_df.index, 
                size=max(1, int(len(stock_df) * 0.1)), 
                replace=False
            )
            
            for idx in indices:
                current_stock = stock_df.loc[idx, 'stock']
                # Random change between -5 and +20 units
                change = np.random.randint(-5, 21)
                new_stock = max(0, current_stock + change)
                stock_df.loc[idx, 'stock'] = new_stock
            
            # Update cache
            realtime_data["stock_data"] = stock_df
            
            # Update database with new stock levels
            try:
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                
                for idx, row in stock_df.iterrows():
                    cursor.execute(
                        "UPDATE drugs SET stock = ? WHERE id = ?",
                        (row['stock'], row['id'])
                    )
                
                conn.commit()
                conn.close()
            except Exception as e:
                logger.error(f"Error updating database with new stock levels: {e}")
            
            last_updates["stock_data"] = datetime.now()
            logger.info(f"Simulated stock updates for {len(indices)} drugs")
            return True
    
    except Exception as e:
        logger.error(f"Error updating stock data: {e}")
    
    return False

def update_sales_data():
    """Update sales data with new transactions or simulate sales"""
    try:
        # If real API configured, use it
        if DATA_SOURCES["sales_updates"]:
            response = requests.get(DATA_SOURCES["sales_updates"], timeout=10)
            if response.status_code == 200:
                new_sales_data = pd.DataFrame(response.json())
                if not new_sales_data.empty:
                    realtime_data["sales_data"] = new_sales_data
                    last_updates["sales_data"] = datetime.now()
                    return True
        
        # Otherwise, simulate new sales
        if realtime_data["stock_data"] is not None and realtime_data["price_data"] is not None:
            # Get current stock and price data
            stock_df = realtime_data["stock_data"].copy()
            price_df = realtime_data["price_data"].copy()
            
            # Merge stock and price data
            merged_df = stock_df.merge(price_df[['id', 'price']], on='id', how='left')
            
            # Generate random sales (1-3 sales transactions)
            num_sales = np.random.randint(1, 4)
            current_date = datetime.now().strftime('%Y-%m-%d')
            
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Get current max sale ID
            cursor.execute("SELECT MAX(id) FROM sales")
            result = cursor.fetchone()
            next_sale_id = (result[0] or 0) + 1
            
            for _ in range(num_sales):
                # Select random drug with stock > 0
                in_stock = merged_df[merged_df['stock'] > 0]
                
                if in_stock.empty:
                    continue
                
                random_idx = np.random.choice(in_stock.index)
                drug = in_stock.loc[random_idx]
                
                # Random quantity (1-3 units)
                quantity = np.random.randint(1, 4)
                quantity = min(quantity, drug['stock'])  # Don't exceed stock
                
                # Calculate total price
                total_price = quantity * drug['price']
                
                # Add to sales table
                cursor.execute(
                    "INSERT INTO sales (id, drug_id, sale_date, quantity, total_price) VALUES (?, ?, ?, ?, ?)",
                    (next_sale_id, drug['id'], current_date, quantity, total_price)
                )
                
                # Update stock
                new_stock = drug['stock'] - quantity
                cursor.execute(
                    "UPDATE drugs SET stock = ? WHERE id = ?",
                    (new_stock, drug['id'])
                )
                
                # Update in-memory data
                stock_df.loc[stock_df['id'] == drug['id'], 'stock'] = new_stock
                
                next_sale_id += 1
            
            conn.commit()
            conn.close()
            
            # Update in-memory stock data
            realtime_data["stock_data"] = stock_df
            
            # Update sales summary data
            conn = sqlite3.connect(DB_PATH)
            sales_data = pd.read_sql("""
                SELECT date(sale_date) as date, SUM(total_price) as daily_sales
                FROM sales
                WHERE date(sale_date) >= date('now', '-30 days')
                GROUP BY date
                ORDER BY date
            """, conn)
            conn.close()
            
            realtime_data["sales_data"] = sales_data
            
            last_updates["sales_data"] = datetime.now()
            logger.info(f"Simulated {num_sales} new sales transactions")
            return True
    
    except Exception as e:
        logger.error(f"Error updating sales data: {e}")
    
    return False

def update_market_data():
    """Update market data from external source or simulate changes"""
    try:
        # If real API configured, use it
        if DATA_SOURCES["market_data"]:
            response = requests.get(DATA_SOURCES["market_data"], timeout=10)
            if response.status_code == 200:
                new_market_data = pd.DataFrame(response.json())
                if not new_market_data.empty:
                    realtime_data["market_data"] = new_market_data
                    last_updates["market_data"] = datetime.now()
                    return True
        
        # Otherwise, simulate market data
        if realtime_data["market_data"] is not None:
            market_df = realtime_data["market_data"].copy()
            
            # Get latest date and value
            latest_date = market_df['date'].max()
            latest_index = market_df.loc[market_df['date'] == latest_date, 'market_index'].iloc[0]
            
            # Add new data point
            new_date = pd.to_datetime(latest_date) + timedelta(days=1)
            # Random change between -2% and +2%
            change = np.random.normal(0, 0.01) * latest_index
            new_index = latest_index + change
            new_volume = np.random.randint(10000, 50000)
            
            # Append new data
            new_row = pd.DataFrame({
                'date': [new_date],
                'market_index': [new_index],
                'volume': [new_volume]
            })
            
            # Keep only last 30 days
            market_df = pd.concat([market_df, new_row]).reset_index(drop=True)
            market_df = market_df.sort_values('date').tail(30).reset_index(drop=True)
            
            # Update cache
            realtime_data["market_data"] = market_df
            last_updates["market_data"] = datetime.now()
            
            logger.info(f"Updated market data with new value: {new_index:.2f}")
            return True
    
    except Exception as e:
        logger.error(f"Error updating market data: {e}")
    
    return False

def update_all_data():
    """Update all data types that need updates"""
    results = {}
    
    # Check and update price data
    if data_needs_update("price_data"):
        results["price_data"] = update_price_data()
    
    # Check and update stock data
    if data_needs_update("stock_data"):
        results["stock_data"] = update_stock_data()
    
    # Check and update sales data
    if data_needs_update("sales_data"):
        results["sales_data"] = update_sales_data()
    
    # Check and update market data
    if data_needs_update("market_data"):
        results["market_data"] = update_market_data()
    
    return results

def background_update_worker():
    """Background worker to process update tasks"""
    global is_worker_running
    
    is_worker_running = True
    logger.info("Starting background update worker")
    
    try:
        while is_worker_running:
            try:
                # Try to get a task with timeout
                task = update_queue.get(timeout=1)
                
                # Process task
                if task == "update_all":
                    update_all_data()
                elif task == "update_price":
                    update_price_data()
                elif task == "update_stock":
                    update_stock_data()
                elif task == "update_sales":
                    update_sales_data()
                elif task == "update_market":
                    update_market_data()
                elif task == "stop":
                    logger.info("Received stop signal for background worker")
                    break
                
                # Mark task as done
                update_queue.task_done()
            
            except queue.Empty:
                # Queue was empty, check if any data needs update
                for data_type in UPDATE_INTERVALS:
                    if data_needs_update(data_type):
                        if data_type == "price_data":
                            update_price_data()
                        elif data_type == "stock_data":
                            update_stock_data()
                        elif data_type == "sales_data":
                            update_sales_data()
                        elif data_type == "market_data":
                            update_market_data()
                
                # Sleep to prevent CPU spinning
                time.sleep(5)
    
    except Exception as e:
        logger.error(f"Error in background worker: {e}")
    
    is_worker_running = False
    logger.info("Background update worker stopped")

def start_background_updates():
    """Start the background update thread if not running"""
    global is_worker_running
    
    if not is_worker_running:
        # Initialize data if needed
        if all(v is None for v in realtime_data.values()):
            initialize_realtime_data()
        
        # Start worker thread
        worker_thread = threading.Thread(target=background_update_worker, daemon=True)
        worker_thread.start()
        logger.info("Background updates started")
        return True
    
    return False

def stop_background_updates():
    """Stop the background update thread"""
    global is_worker_running
    
    if is_worker_running:
        # Add stop signal to queue
        update_queue.put("stop")
        logger.info("Sent stop signal to background worker")
        return True
    
    return False

def get_realtime_data(data_type):
    """Get real-time data by type, with on-demand updates if needed"""
    # Check if data is available
    if realtime_data.get(data_type) is None:
        # Initialize if not done yet
        initialize_realtime_data()
    
    # Check if data needs update
    if data_needs_update(data_type):
        # Queue update in background
        update_queue.put(f"update_{data_type.split('_')[0]}")
    
    return realtime_data.get(data_type)

def force_data_update(data_type=None):
    """Force immediate update of specified data type or all data"""
    if data_type is None or data_type == "all":
        # Update all data
        return update_all_data()
    elif data_type in UPDATE_INTERVALS:
        # Update specific data type
        if data_type == "price_data":
            return update_price_data()
        elif data_type == "stock_data":
            return update_stock_data()
        elif data_type == "sales_data":
            return update_sales_data()
        elif data_type == "market_data":
            return update_market_data()
    
    return False

def get_update_status():
    """Get the status of data updates"""
    status = {}
    
    for data_type in UPDATE_INTERVALS:
        last_update = last_updates.get(data_type, datetime.min)
        next_update = last_update + timedelta(seconds=UPDATE_INTERVALS[data_type])
        
        status[data_type] = {
            "last_update": last_update.strftime("%Y-%m-%d %H:%M:%S"),
            "next_update": next_update.strftime("%Y-%m-%d %H:%M:%S"),
            "seconds_to_next_update": max(0, (next_update - datetime.now()).total_seconds()),
            "data_available": realtime_data.get(data_type) is not None
        }
    
    status["worker_running"] = is_worker_running
    
    return status

def show_realtime_status():
    """Display real-time data status in Streamlit"""
    status = get_update_status()
    
    st.subheader("Real-Time Data Status")
    
    # Worker status
    if status["worker_running"]:
        st.success("✅ Real-time data updates are active")
    else:
        st.warning("⚠️ Real-time data updates are not running")
        if st.button("Start Real-Time Updates"):
            if start_background_updates():
                st.success("Real-time updates started!")
                # Rerun to update status
                st.experimental_rerun()
    
    # Data status
    for data_type, data_status in status.items():
        if data_type != "worker_running":
            col1, col2, col3 = st.columns(3)
            
            with col1:
                name = data_type.replace("_", " ").title()
                if data_status["data_available"]:
                    st.markdown(f"**{name}:** Available")
                else:
                    st.markdown(f"**{name}:** Not Available")
            
            with col2:
                st.markdown(f"**Last Update:** {data_status['last_update']}")
            
            with col3:
                seconds = data_status["seconds_to_next_update"]
                if seconds <= 0:
                    st.markdown("**Next Update:** Due now")
                else:
                    minutes = int(seconds // 60)
                    secs = int(seconds % 60)
                    st.markdown(f"**Next Update:** {minutes}m {secs}s")
    
    # Manual update button
    if st.button("Force Update All Data"):
        with st.spinner("Updating all data..."):
            results = force_data_update()
            
            success_count = sum(1 for r in results.values() if r)
            if success_count > 0:
                st.success(f"Successfully updated {success_count} data types")
            else:
                st.warning("No data was updated. Check logs for details.")

def on_app_start():
    """Initialize real-time data when the app starts"""
    initialize_realtime_data()
    start_background_updates()

def on_app_stop():
    """Clean up when the app stops"""
    stop_background_updates()

if __name__ == "__main__":
    # Test functionality
    print("Initializing real-time data...")
    initialize_realtime_data()
    
    print("Starting background updates...")
    start_background_updates()
    
    print("Waiting for updates...")
    time.sleep(10)
    
    print("Forcing update...")
    results = force_data_update()
    print(f"Update results: {results}")
    
    print("Getting status...")
    status = get_update_status()
    print(json.dumps(status, default=str, indent=2))
    
    print("Stopping background updates...")
    stop_background_updates()
    
    print("Done.")