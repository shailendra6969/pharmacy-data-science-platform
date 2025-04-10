"""
Gene Data Integration module for the Pharmacy Data Science Platform.

This module provides functions to fetch, process, and store gene-drug interaction data
from various sources including PharmGKB, DrugBank, and local datasets.
"""
import os
import json
import requests
import pandas as pd
import time
import random
import hashlib
from datetime import datetime, timedelta
import concurrent.futures
from config import DATA_DIR, logger, API_TIMEOUT, API_RETRY_ATTEMPTS

# PharmGKB API endpoints
PHARMGKB_BASE_URL = "https://api.pharmgkb.org/v1/data/"
PHARMGKB_GENE_URL = PHARMGKB_BASE_URL + "gene/"
PHARMGKB_DRUG_URL = PHARMGKB_BASE_URL + "chemical/"
PHARMGKB_VARIANT_URL = PHARMGKB_BASE_URL + "variant/"

# DrugBank API endpoints (would require registration and API key in production)
DRUGBANK_API_URL = "https://go.drugbank.com/api/v1/"

# Local cache settings
CACHE_DIR = os.path.join(DATA_DIR, "gene_cache")
CACHE_EXPIRY_DAYS = 30  # Cache validity period

def ensure_cache_dir():
    """Ensure the cache directory exists"""
    os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_key(url, params=None):
    """Generate a unique cache key for a request"""
    if params:
        key_data = url + str(sorted(params.items()))
    else:
        key_data = url
    return hashlib.md5(key_data.encode()).hexdigest()

def is_cache_valid(cache_file):
    """Check if a cache file is still valid based on expiry time"""
    if not os.path.exists(cache_file):
        return False
    
    file_modified_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
    cache_expiry = datetime.now() - timedelta(days=CACHE_EXPIRY_DAYS)
    
    return file_modified_time > cache_expiry

def get_from_cache(cache_key):
    """Retrieve data from cache if available and valid"""
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    
    if is_cache_valid(cache_file):
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading from cache: {e}")
    
    return None

def save_to_cache(cache_key, data):
    """Save API response data to cache"""
    ensure_cache_dir()
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    
    try:
        with open(cache_file, 'w') as f:
            json.dump(data, f)
        return True
    except Exception as e:
        logger.error(f"Error saving to cache: {e}")
        return False

def api_request(url, params=None, use_cache=True):
    """Make an API request with caching, timeout, and retry logic"""
    # Check cache first if enabled
    if use_cache:
        cache_key = get_cache_key(url, params)
        cached_data = get_from_cache(cache_key)
        if cached_data:
            logger.debug(f"Retrieved data from cache for {url}")
            return cached_data, True  # Second value indicates cache hit
    
    # Make API request with retry logic
    for attempt in range(API_RETRY_ATTEMPTS):
        try:
            response = requests.get(url, params=params, timeout=API_TIMEOUT)
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    
                    # Save to cache if successful and caching is enabled
                    if use_cache:
                        cache_key = get_cache_key(url, params)
                        save_to_cache(cache_key, data)
                    
                    return data, False  # Second value indicates no cache hit
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON response from {url}")
            
            elif response.status_code == 404:
                logger.warning(f"Resource not found: {url}")
                return None, False
            
            elif response.status_code == 429:
                # Rate limiting - exponential backoff
                wait_time = (2 ** attempt) + random.random()
                logger.warning(f"Rate limited, waiting {wait_time:.2f}s before retry")
                time.sleep(wait_time)
                continue
            
            else:
                logger.warning(f"API request failed with status {response.status_code}: {url}")
        
        except requests.exceptions.Timeout:
            logger.warning(f"Request timed out for {url} (attempt {attempt+1}/{API_RETRY_ATTEMPTS})")
        
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request error for {url}: {e}")
        
        # Add delay before retry
        if attempt < API_RETRY_ATTEMPTS - 1:
            time.sleep(1 * (attempt + 1))
    
    logger.error(f"Failed to fetch data after {API_RETRY_ATTEMPTS} attempts: {url}")
    return None, False

def get_gene_data(gene_id):
    """Fetch detailed gene data from PharmGKB API with caching"""
    url = PHARMGKB_GENE_URL + gene_id
    return api_request(url)

def get_drug_data(drug_id):
    """Fetch detailed drug data from PharmGKB API with caching"""
    url = PHARMGKB_DRUG_URL + drug_id
    return api_request(url)

def get_variant_data(variant_id):
    """Fetch detailed variant data from PharmGKB API with caching"""
    url = PHARMGKB_VARIANT_URL + variant_id
    return api_request(url)

def get_gene_drug_interactions(gene_id):
    """Extract drug interactions for a specific gene from PharmGKB"""
    gene_data, cache_hit = get_gene_data(gene_id)
    interactions = []
    
    if gene_data:
        gene_name = gene_data.get('name', 'Unknown')
        related_chemicals = gene_data.get('relatedChemicals', [])
        
        for chemical in related_chemicals:
            interactions.append({
                "Gene": gene_name,
                "Gene ID": gene_id,
                "Drug": chemical.get('name', 'Unknown Drug'),
                "Drug ID": chemical.get('id', ''),
                "Relation": chemical.get('relation', 'Unknown'),
                "Evidence": chemical.get('evidenceLevel', 'Unknown'),
                "Source": "PharmGKB"
            })
    
    return interactions, gene_data is not None

def batch_fetch_gene_interactions(gene_ids, max_workers=5):
    """Fetch gene-drug interactions for multiple genes in parallel"""
    all_interactions = []
    success_count = 0
    failed_genes = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_gene = {executor.submit(get_gene_drug_interactions, gene_id): gene_id for gene_id in gene_ids}
        
        for future in concurrent.futures.as_completed(future_to_gene):
            gene_id = future_to_gene[future]
            try:
                interactions, success = future.result()
                if success:
                    all_interactions.extend(interactions)
                    success_count += 1
                else:
                    failed_genes.append(gene_id)
            except Exception as e:
                logger.error(f"Error processing gene {gene_id}: {e}")
                failed_genes.append(gene_id)
    
    return all_interactions, success_count, failed_genes

def create_interaction_network(interactions):
    """Create a network representation from gene-drug interactions"""
    if not interactions:
        return None
    
    # Convert to DataFrame for easier processing
    df = pd.DataFrame(interactions)
    
    # Create network data structure
    network = {
        "nodes": [],
        "edges": []
    }
    
    # Process genes
    genes = df[['Gene', 'Gene ID']].drop_duplicates()
    for _, row in genes.iterrows():
        network["nodes"].append({
            "id": row['Gene ID'],
            "name": row['Gene'],
            "type": "gene"
        })
    
    # Process drugs
    drugs = df[['Drug', 'Drug ID']].drop_duplicates()
    for _, row in drugs.iterrows():
        # Skip entries with missing Drug ID
        if pd.isna(row['Drug ID']) or not row['Drug ID']:
            continue
            
        network["nodes"].append({
            "id": row['Drug ID'],
            "name": row['Drug'],
            "type": "drug"
        })
    
    # Create edges
    for _, row in df.iterrows():
        # Skip entries with missing Drug ID
        if pd.isna(row['Drug ID']) or not row['Drug ID']:
            continue
            
        network["edges"].append({
            "source": row['Gene ID'],
            "target": row['Drug ID'],
            "relation": row['Relation'],
            "evidence": row.get('Evidence', "Unknown")
        })
    
    return network

def generate_simulation_gene_data(gene_id):
    """Generate simulated gene data for demonstration purposes"""
    
    # Check if gene ID follows PharmGKB format (e.g., PA12345)
    is_valid_format = len(gene_id) >= 4 and gene_id.startswith("PA")
    
    # Common gene names with their functions
    gene_names = {
        "PA124": "CYP2D6",
        "PA128": "CYP3A4",
        "PA130": "CYP2C9",
        "PA131": "CYP2C19",
        "PA134": "CYP1A2",
        "PA151": "VKORC1",
        "PA159": "DPYD",
        "PA162": "TPMT",
        "PA166": "SLCO1B1",
        "PA172": "UGT1A1",
        "PA182": "COMT",
        "PA198": "ABCB1",
        "PA200": "OPRM1"
    }
    
    # Drug categories and examples
    drug_categories = {
        "Cardiovascular": ["Warfarin", "Clopidogrel", "Simvastatin", "Atorvastatin", "Metoprolol"],
        "Psychiatric": ["Fluoxetine", "Sertraline", "Citalopram", "Amitriptyline", "Haloperidol"],
        "Pain": ["Codeine", "Tramadol", "Oxycodone", "Morphine", "Fentanyl"],
        "Oncology": ["Tamoxifen", "Capecitabine", "Fluorouracil", "Irinotecan", "Mercaptopurine"],
        "Infectious": ["Voriconazole", "Efavirenz", "Isoniazid", "Abacavir", "Atovaquone"]
    }
    
    # Generate gene name
    if gene_id in gene_names:
        gene_name = gene_names[gene_id]
    elif is_valid_format:
        # Generate consistent name for valid-looking IDs
        seed = int(gene_id[2:]) if gene_id[2:].isdigit() else hash(gene_id)
        random.seed(seed)
        prefixes = ["CYP", "HLA", "UGT", "DPYD", "TPMT", "VKORC", "SLCO", "ABC", "DRD", "HTR"]
        suffix = f"{random.randint(1, 9)}{random.choice('ABCD')}{random.randint(1, 9)}"
        gene_name = f"{random.choice(prefixes)}{suffix}"
    else:
        gene_name = f"Gene-{gene_id}"
    
    # Generate related drugs
    related_drugs = []
    
    # Number of related drugs
    seed = hash(gene_id) if isinstance(gene_id, str) else gene_id
    random.seed(seed)
    num_drugs = random.randint(0, 5)
    
    # Relationship types
    relationships = ["substrate", "inhibitor", "inducer", "target", "metabolizer"]
    
    # Evidence levels
    evidence_levels = ["high", "moderate", "low"]
    
    # Generate drug interactions
    for _ in range(num_drugs):
        # Select category and drug
        category = random.choice(list(drug_categories.keys()))
        drug = random.choice(drug_categories[category])
        
        # Generate unique ID
        drug_id = f"PA{random.randint(10000, 99999)}"
        
        related_drugs.append({
            "id": drug_id,
            "name": drug,
            "relation": random.choice(relationships),
            "evidenceLevel": random.choice(evidence_levels),
            "category": category
        })
    
    # Create simulated gene data
    simulated_data = {
        "id": gene_id,
        "name": gene_name,
        "symbol": gene_name,
        "relatedChemicals": related_drugs,
        "chromosomeLocation": f"chr{random.randint(1, 22)}:{random.randint(1000000, 100000000)}",
        "source": "simulation"
    }
    
    return simulated_data

def fetch_gene_data(gene_ids, use_simulation=False):
    """Fetch gene data and interactions from API or simulation"""
    
    # Store results
    gene_data_results = []
    all_interactions = []
    success_count = 0
    failed_genes = []
    
    # Process each gene ID
    for gene_id in gene_ids:
        gene_id = gene_id.strip()
        if not gene_id:
            continue
        
        if use_simulation:
            # Generate simulated data
            simulated_data = generate_simulation_gene_data(gene_id)
            gene_data_results.append(simulated_data)
            
            # Extract interactions
            gene_name = simulated_data.get('name', 'Unknown')
            related_chemicals = simulated_data.get('relatedChemicals', [])
            
            for chemical in related_chemicals:
                all_interactions.append({
                    "Gene": gene_name,
                    "Gene ID": gene_id,
                    "Drug": chemical.get('name', 'Unknown Drug'),
                    "Drug ID": chemical.get('id', ''),
                    "Relation": chemical.get('relation', 'Unknown'),
                    "Evidence": chemical.get('evidenceLevel', 'Unknown'),
                    "Category": chemical.get('category', 'Unknown'),
                    "Source": "Simulation"
                })
            
            success_count += 1
            
        else:
            # Use real API with cache
            interactions, success = get_gene_drug_interactions(gene_id)
            
            if success:
                all_interactions.extend(interactions)
                success_count += 1
            else:
                failed_genes.append(gene_id)
    
    return all_interactions, gene_data_results, success_count, failed_genes

def create_default_interaction_dataset():
    """Create and save a comprehensive default gene-drug interaction dataset"""
    
    # Common pharmacogenomic genes
    genes = [
        {"id": "PA124", "name": "CYP2D6"},
        {"id": "PA128", "name": "CYP3A4"},
        {"id": "PA130", "name": "CYP2C9"},
        {"id": "PA131", "name": "CYP2C19"},
        {"id": "PA134", "name": "CYP1A2"},
        {"id": "PA151", "name": "VKORC1"},
        {"id": "PA159", "name": "DPYD"},
        {"id": "PA162", "name": "TPMT"},
        {"id": "PA166", "name": "SLCO1B1"},
        {"id": "PA172", "name": "UGT1A1"}
    ]
    
    # Important drugs with known pharmacogenomic interactions
    drugs = [
        {"id": "PA450704", "name": "Warfarin", "category": "Cardiovascular"},
        {"id": "PA449053", "name": "Clopidogrel", "category": "Cardiovascular"},
        {"id": "PA451363", "name": "Simvastatin", "category": "Cardiovascular"},
        {"id": "PA448031", "name": "Codeine", "category": "Pain"},
        {"id": "PA449509", "name": "Tramadol", "category": "Pain"},
        {"id": "PA449383", "name": "Tamoxifen", "category": "Oncology"},
        {"id": "PA448771", "name": "Fluorouracil", "category": "Oncology"},
        {"id": "PA448785", "name": "Mercaptopurine", "category": "Oncology"},
        {"id": "PA451906", "name": "Voriconazole", "category": "Infectious"},
        {"id": "PA449005", "name": "Efavirenz", "category": "Infectious"}
    ]
    
    # Gene-drug interactions based on clinical guidelines
    interactions = [
        # CYP2D6 interactions
        {"gene_id": "PA124", "drug_id": "PA448031", "relation": "metabolism", "evidence": "high", 
         "effect": "Reduced codeine efficacy in poor metabolizers", "recommendation": "Consider alternative analgesic"},
        {"gene_id": "PA124", "drug_id": "PA449509", "relation": "metabolism", "evidence": "moderate", 
         "effect": "Altered tramadol efficacy", "recommendation": "Monitor response"},
        {"gene_id": "PA124", "drug_id": "PA449383", "relation": "metabolism", "evidence": "high", 
         "effect": "Reduced conversion to active metabolite", "recommendation": "Consider alternative therapy"},
        
        # CYP2C19 interactions
        {"gene_id": "PA131", "drug_id": "PA449053", "relation": "metabolism", "evidence": "high", 
         "effect": "Reduced antiplatelet effect in poor metabolizers", "recommendation": "Consider alternative antiplatelet"},
        {"gene_id": "PA131", "drug_id": "PA451906", "relation": "metabolism", "evidence": "moderate", 
         "effect": "Increased drug exposure", "recommendation": "Consider dose reduction"},
        
        # CYP3A4 interactions
        {"gene_id": "PA128", "drug_id": "PA449005", "relation": "metabolism", "evidence": "moderate", 
         "effect": "Altered drug levels", "recommendation": "Monitor response"},
        {"gene_id": "PA128", "drug_id": "PA451363", "relation": "metabolism", "evidence": "high", 
         "effect": "Increased risk of myopathy", "recommendation": "Lower dose"},
        
        # CYP2C9 interactions
        {"gene_id": "PA130", "drug_id": "PA450704", "relation": "metabolism", "evidence": "high", 
         "effect": "Altered anticoagulant response", "recommendation": "Adjust dose based on genotype"},
        
        # VKORC1 interactions
        {"gene_id": "PA151", "drug_id": "PA450704", "relation": "target", "evidence": "high", 
         "effect": "Altered warfarin sensitivity", "recommendation": "Adjust dose based on genotype"},
        
        # DPYD interactions
        {"gene_id": "PA159", "drug_id": "PA448771", "relation": "metabolism", "evidence": "high", 
         "effect": "Increased toxicity risk", "recommendation": "Consider alternative or reduced dose"},
        
        # TPMT interactions
        {"gene_id": "PA162", "drug_id": "PA448785", "relation": "metabolism", "evidence": "high", 
         "effect": "Increased toxicity risk", "recommendation": "Reduce dose or consider alternative"},
        
        # SLCO1B1 interactions
        {"gene_id": "PA166", "drug_id": "PA451363", "relation": "transport", "evidence": "high", 
         "effect": "Increased risk of myopathy", "recommendation": "Consider lower dose or alternative statin"},
    ]
    
    # Build complete interaction dataset
    dataset = []
    
    for interaction in interactions:
        gene = next((g for g in genes if g["id"] == interaction["gene_id"]), {"id": interaction["gene_id"], "name": "Unknown"})
        drug = next((d for d in drugs if d["id"] == interaction["drug_id"]), {"id": interaction["drug_id"], "name": "Unknown", "category": "Unknown"})
        
        dataset.append({
            "Gene": gene["name"],
            "Gene ID": gene["id"],
            "Drug": drug["name"],
            "Drug ID": drug["id"],
            "Category": drug.get("category", "Unknown"),
            "Relation": interaction["relation"],
            "Evidence": interaction["evidence"],
            "Effect": interaction.get("effect", ""),
            "Recommendation": interaction.get("recommendation", ""),
            "Source": "Default Dataset"
        })
    
    # Save the dataset
    output_file = os.path.join(DATA_DIR, "default_gene_interactions.csv")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    df = pd.DataFrame(dataset)
    df.to_csv(output_file, index=False)
    
    logger.info(f"Created default gene interaction dataset with {len(dataset)} interactions")
    
    return dataset

def load_default_interaction_dataset():
    """Load the default gene-drug interaction dataset"""
    file_path = os.path.join(DATA_DIR, "default_gene_interactions.csv")
    
    if not os.path.exists(file_path):
        logger.info("Default gene interaction dataset not found, creating it")
        dataset = create_default_interaction_dataset()
        return dataset
    
    try:
        df = pd.read_csv(file_path)
        return df.to_dict('records')
    except Exception as e:
        logger.error(f"Error loading default gene interaction dataset: {e}")
        return []

def main():
    """Test the gene data integration functionality"""
    gene_ids = ["PA124", "PA128", "PA130", "PA131", "PA134", "PA151"]
    
    print(f"Fetching data for {len(gene_ids)} genes using simulation...")
    interactions, gene_data, success_count, failed_genes = fetch_gene_data(gene_ids, use_simulation=True)
    
    print(f"Successfully fetched data for {success_count} genes")
    if failed_genes:
        print(f"Failed to fetch data for {len(failed_genes)} genes: {failed_genes}")
    
    print(f"Found {len(interactions)} gene-drug interactions")
    
    if interactions:
        # Display first few interactions
        df = pd.DataFrame(interactions)
        print("\nSample interactions:")
        print(df.head())
        
        # Create and display network
        network = create_interaction_network(interactions)
        if network:
            print(f"\nNetwork created with {len(network['nodes'])} nodes and {len(network['edges'])} edges")

if __name__ == "__main__":
    main()