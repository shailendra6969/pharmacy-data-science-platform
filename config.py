"""
Configuration settings for the Pharmacy Data Science Platform.
"""
import os
import logging

# App information
APP_TITLE = "Pharmacy Data Science Platform"
APP_VERSION = "2.0.0"

# Directory settings
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")
CACHE_DIR = os.path.join(DATA_DIR, "cache")

# Create directories if they don't exist
for directory in [DATA_DIR, MODEL_DIR, LOG_DIR, CACHE_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Database configuration
DB_PATH = os.path.join(DATA_DIR, "pharmacy_data.db")
CSV_PATH = os.path.join(DATA_DIR, "sample_100_drugs.csv")
COMPREHENSIVE_CSV_PATH = os.path.join(DATA_DIR, "comprehensive_drugs.csv")

# API Configuration
API_TIMEOUT = 10  # seconds
API_RETRY_ATTEMPTS = 3

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "app.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# MongoDB settings
MONGO_URI = "mongodb://localhost:27017/"
MONGO_DB = "pharmacy_db"
MONGO_CONN_TIMEOUT = 5000  # ms

# Caching settings
CACHE_EXPIRY_DAYS = 30

# Real-time update settings
ENABLE_REALTIME_UPDATES = True
REALTIME_UPDATE_INTERVAL = 300  # seconds (5 minutes)

# External API endpoints
RXNORM_API_URL = "https://rxnav.nlm.nih.gov/REST/rxcui.json"
FDA_API_URL = "https://api.fda.gov/drug/ndc.json"
DAILYMED_API_URL = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls"
PHARMGKB_API_URL = "https://api.pharmgkb.org/v1/data/"

# List of available modules
MODULES = [
    "Dashboard",
    "Gene–Drug Analysis",
    "Sales Forecast",
    "Drug Price ML Model",
    "Drug Shortage Prediction",
    "Medicine Recommendation", 
    "Gene Interaction Network",
    "Product Catalog",
    "SQLite Database Explorer",
    "MongoDB Integration",
    "Medicine Verification",
    "Documentation"
]

# Feature flags
FEATURES = {
    "enable_real_time_updates": True,
    "enable_mongodb_integration": True,
    "enable_comprehensive_drug_data": True,
    "enable_gene_interaction_simulation": True,
    "enable_medicine_verification": True,
    "enable_external_apis": False  # Set to True to use real external APIs
}