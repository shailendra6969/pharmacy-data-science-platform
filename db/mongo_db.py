"""
MongoDB integration for the Pharmacy Data Science Platform.
"""
import os
import json
import time
import logging
from datetime import datetime
from pymongo import MongoClient, errors
from config import MONGO_URI, MONGO_DB, MONGO_CONN_TIMEOUT, logger

class MongoDBHandler:
    """Handler for MongoDB operations with connection pooling and retry logic"""
    
    _instance = None
    _client = None
    _db = None
    
    @classmethod
    def get_instance(cls):
        """Singleton pattern to reuse connection"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        """Initialize the MongoDB handler"""
        self.connect()
    
    def connect(self):
        """Establish connection to MongoDB with retry mechanism"""
        max_retries = 3
        retry_delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
                self._client = MongoClient(
                    MONGO_URI, 
                    serverSelectionTimeoutMS=MONGO_CONN_TIMEOUT,
                    connectTimeoutMS=MONGO_CONN_TIMEOUT,
                    socketTimeoutMS=MONGO_CONN_TIMEOUT * 2,
                    maxPoolSize=10,
                    minPoolSize=1,
                    maxIdleTimeMS=45000,
                    waitQueueTimeoutMS=MONGO_CONN_TIMEOUT
                )
                
                # Test connection
                self._client.server_info()
                
                # Connect to database
                self._db = self._client[MONGO_DB]
                
                logger.info("Successfully connected to MongoDB")
                return True
                
            except errors.ServerSelectionTimeoutError:
                logger.error(f"MongoDB server selection timeout (attempt {attempt+1}/{max_retries})")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error("Failed to connect to MongoDB after multiple attempts")
                    return False
            
            except errors.ConnectionFailure:
                logger.error(f"MongoDB connection failure (attempt {attempt+1}/{max_retries})")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logger.error("Failed to connect to MongoDB after multiple attempts")
                    return False
            
            except Exception as e:
                logger.error(f"MongoDB connection error: {e}")
                return False
    
    def is_connected(self):
        """Check if connection to MongoDB is active"""
        if self._client is None:
            return False
        
        try:
            # Test connection by executing a simple command
            self._client.admin.command('ping')
            return True
        except Exception:
            return False
    
    def reconnect_if_needed(self):
        """Reconnect to MongoDB if connection is lost"""
        if not self.is_connected():
            logger.info("MongoDB connection lost, attempting to reconnect...")
            return self.connect()
        return True
    
    def get_db(self):
        """Get the database instance"""
        if self.reconnect_if_needed():
            return self._db
        return None
    
    def get_collection(self, collection_name):
        """Get a collection by name"""
        db = self.get_db()
        if db:
            return db[collection_name]
        return None
    
    def list_collections(self):
        """List all collections in the database"""
        db = self.get_db()
        if db:
            try:
                return db.list_collection_names()
            except Exception as e:
                logger.error(f"Error listing collections: {e}")
        return []
    
    def create_collection(self, collection_name):
        """Create a new collection"""
        db = self.get_db()
        if db:
            try:
                db.create_collection(collection_name)
                logger.info(f"Collection '{collection_name}' created successfully")
                return True
            except errors.CollectionInvalid:
                logger.warning(f"Collection '{collection_name}' already exists")
                return True
            except Exception as e:
                logger.error(f"Error creating collection: {e}")
        return False
    
    def insert_document(self, collection_name, document):
        """Insert a single document into a collection"""
        collection = self.get_collection(collection_name)
        if collection:
            try:
                # Add timestamp if not present
                if 'timestamp' not in document:
                    document['timestamp'] = datetime.now().isoformat()
                
                result = collection.insert_one(document)
                return result.inserted_id
            except Exception as e:
                logger.error(f"Error inserting document: {e}")
        return None
    
    def insert_documents(self, collection_name, documents):
        """Insert multiple documents into a collection"""
        collection = self.get_collection(collection_name)
        if collection:
            try:
                # Add timestamp if not present
                for doc in documents:
                    if 'timestamp' not in doc:
                        doc['timestamp'] = datetime.now().isoformat()
                
                result = collection.insert_many(documents)
                return result.inserted_ids
            except Exception as e:
                logger.error(f"Error inserting documents: {e}")
        return []
    
    def find_documents(self, collection_name, query=None, limit=None):
        """Find documents in a collection matching the query"""
        collection = self.get_collection(collection_name)
        if collection:
            try:
                if query is None:
                    query = {}
                
                cursor = collection.find(query)
                
                if limit:
                    cursor = cursor.limit(limit)
                
                return list(cursor)
            except Exception as e:
                logger.error(f"Error finding documents: {e}")
        return []
    
    def count_documents(self, collection_name, query=None):
        """Count documents in a collection matching the query"""
        collection = self.get_collection(collection_name)
        if collection:
            try:
                if query is None:
                    query = {}
                
                return collection.count_documents(query)
            except Exception as e:
                logger.error(f"Error counting documents: {e}")
        return 0
    
    def aggregate(self, collection_name, pipeline):
        """Execute an aggregation pipeline on a collection"""
        collection = self.get_collection(collection_name)
        if collection:
            try:
                return list(collection.aggregate(pipeline))
            except Exception as e:
                logger.error(f"Error executing aggregation: {e}")
        return []
    
    def update_document(self, collection_name, query, update):
        """Update a single document in a collection"""
        collection = self.get_collection(collection_name)
        if collection:
            try:
                result = collection.update_one(query, update)
                return result.modified_count
            except Exception as e:
                logger.error(f"Error updating document: {e}")
        return 0
    
    def delete_document(self, collection_name, query):
        """Delete a single document from a collection"""
        collection = self.get_collection(collection_name)
        if collection:
            try:
                result = collection.delete_one(query)
                return result.deleted_count
            except Exception as e:
                logger.error(f"Error deleting document: {e}")
        return 0
    
    def export_collection(self, collection_name):
        """Export all documents from a collection"""
        documents = self.find_documents(collection_name)
        if documents:
            return documents
        return []
    
    def close(self):
        """Close the MongoDB connection"""
        if self._client:
            try:
                self._client.close()
                logger.info("MongoDB connection closed")
            except Exception as e:
                logger.error(f"Error closing MongoDB connection: {e}")

# Global instance getter
def get_mongodb_handler():
    """Get the MongoDB handler instance"""
    return MongoDBHandler.get_instance()