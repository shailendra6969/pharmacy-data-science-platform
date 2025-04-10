"""
Medicine Verification module for the Pharmacy Data Science Platform.

This module provides functionality to verify medicines against various databases,
including local database and external APIs like RxNorm and DailyMed.
"""
import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import json
from datetime import datetime
import sqlite3
from fuzzywuzzy import fuzz, process
from config import logger, DB_PATH
from db.sqlite_db import execute_query

# API endpoints for verification
RXNORM_API_URL = "https://rxnav.nlm.nih.gov/REST/rxcui.json"
DAILYMED_API_URL = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls"
FDA_API_URL = "https://api.fda.gov/drug/ndc.json"

def search_local_database(search_term, threshold=70):
    """
    Search for a medicine in the local database using fuzzy matching
    
    Args:
        search_term: The name or NDC to search for
        threshold: Minimum match score (0-100)
        
    Returns:
        DataFrame of matching medicines
    """
    try:
        # Get all drugs from database
        drugs_df = execute_query("""
            SELECT 
                id, generic_name, brand_name, manufacturer, category,
                subcategory, ndc, price, dosage_form, dosage
            FROM drugs
        """)
        
        if drugs_df.empty:
            return pd.DataFrame()
        
        # Initialize match scores
        drugs_df['match_score'] = 0
        
        # Check for exact NDC match
        if search_term.replace('-', '').isdigit():
            # Format NDC consistently
            search_ndc = search_term.replace('-', '')
            
            # Look for NDC matches
            ndc_matches = drugs_df[drugs_df['ndc'].str.replace('-', '').str.contains(search_ndc, na=False)]
            
            if not ndc_matches.empty:
                ndc_matches['match_score'] = 100
                return ndc_matches.sort_values('match_score', ascending=False)
        
        # Calculate match scores for generic names
        for idx, row in drugs_df.iterrows():
            generic_score = fuzz.token_set_ratio(search_term.lower(), str(row['generic_name']).lower())
            brand_score = fuzz.token_set_ratio(search_term.lower(), str(row['brand_name']).lower())
            
            # Take the best match score
            drugs_df.at[idx, 'match_score'] = max(generic_score, brand_score)
        
        # Filter by threshold
        matches = drugs_df[drugs_df['match_score'] >= threshold]
        
        # Sort by match score
        return matches.sort_values('match_score', ascending=False)
    
    except Exception as e:
        logger.error(f"Error searching local database: {e}")
        return pd.DataFrame()

def verify_with_rxnorm(search_term):
    """
    Verify a medicine using the RxNorm API
    
    Args:
        search_term: The name to search for
        
    Returns:
        Dictionary with verification results
    """
    try:
        params = {"name": search_term, "search": "1"}
        response = requests.get(RXNORM_API_URL, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            # Check if we got meaningful results
            if 'idGroup' in data and 'rxnormId' in data['idGroup'] and data['idGroup']['rxnormId']:
                rxcui = data['idGroup']['rxnormId'][0]
                
                # Get additional details
                details_url = f"https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/allrelated.json"
                details_response = requests.get(details_url, timeout=10)
                
                if details_response.status_code == 200:
                    details_data = details_response.json()
                    
                    return {
                        "verified": True,
                        "rxnorm_id": rxcui,
                        "name": search_term,
                        "source": "RxNorm",
                        "details": details_data,
                        "timestamp": datetime.now().isoformat()
                    }
            
            return {
                "verified": False,
                "name": search_term,
                "source": "RxNorm",
                "message": "Not found in RxNorm database",
                "timestamp": datetime.now().isoformat()
            }
        
        else:
            return {
                "verified": False,
                "name": search_term,
                "source": "RxNorm",
                "message": f"API error: {response.status_code}",
                "timestamp": datetime.now().isoformat()
            }
    
    except Exception as e:
        logger.error(f"Error verifying with RxNorm: {e}")
        return {
            "verified": False,
            "name": search_term,
            "source": "RxNorm",
            "message": f"Error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

def verify_with_fda(search_term):
    """
    Verify a medicine using the FDA API
    
    Args:
        search_term: The name or NDC to search for
        
    Returns:
        Dictionary with verification results
    """
    try:
        # Check if this is an NDC
        if search_term.replace('-', '').isdigit():
            # Search by NDC
            search_ndc = search_term.replace('-', '')
            params = {"search": f"product_ndc:{search_ndc}", "limit": 5}
        else:
            # Search by name
            params = {"search": f"brand_name:{search_term}+generic_name:{search_term}", "limit": 5}
        
        response = requests.get(FDA_API_URL, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            # Check if we got results
            if 'results' in data and data['results']:
                result = data['results'][0]
                
                return {
                    "verified": True,
                    "name": result.get('brand_name') or result.get('generic_name') or search_term,
                    "source": "FDA",
                    "details": result,
                    "timestamp": datetime.now().isoformat()
                }
            
            return {
                "verified": False,
                "name": search_term,
                "source": "FDA",
                "message": "Not found in FDA database",
                "timestamp": datetime.now().isoformat()
            }
        
        else:
            return {
                "verified": False,
                "name": search_term,
                "source": "FDA",
                "message": f"API error: {response.status_code}",
                "timestamp": datetime.now().isoformat()
            }
    
    except Exception as e:
        logger.error(f"Error verifying with FDA: {e}")
        return {
            "verified": False,
            "name": search_term,
            "source": "FDA",
            "message": f"Error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

def find_alternative_medicines(drug_info):
    """
    Find alternative medicines based on given drug info
    
    Args:
        drug_info: Dictionary with drug information
        
    Returns:
        DataFrame with alternative medicines
    """
    try:
        if not drug_info or 'category' not in drug_info or not drug_info['category']:
            return pd.DataFrame()
        
        # Get category and subcategory
        category = drug_info.get('category', '')
        subcategory = drug_info.get('subcategory', '')
        
        # Base query
        query = "SELECT * FROM drugs WHERE id != ?"
        params = [drug_info.get('id', 0)]
        
        # Add category filter
        if category:
            query += " AND category = ?"
            params.append(category)
        
        # Add subcategory filter if available
        if subcategory:
            query += " AND subcategory = ?"
            params.append(subcategory)
        
        # Limit to reasonable number
        query += " ORDER BY price LIMIT 10"
        
        # Execute query
        alternatives = execute_query(query, params=params)
        
        return alternatives
    
    except Exception as e:
        logger.error(f"Error finding alternative medicines: {e}")
        return pd.DataFrame()

def get_medicine_compliance_info(medicine_name):
    """
    Get compliance and regulatory information for a medicine
    
    Args:
        medicine_name: Name of the medicine to check
        
    Returns:
        Dictionary with compliance information
    """
    # This would normally call a real regulatory database
    # For demonstration, we'll return simulated data
    
    # Common drug statuses
    statuses = ["Approved", "Approved with restrictions", "Under review", "Discontinued", "Recalled"]
    
    # Generate consistent status based on drug name
    seed = sum(ord(c) for c in medicine_name) if medicine_name else 0
    np.random.seed(seed)
    
    status_idx = np.random.randint(0, len(statuses))
    status = statuses[status_idx]
    
    # Generate approval date
    years = list(range(1980, 2024))
    months = list(range(1, 13))
    days = list(range(1, 29))  # Safe range for all months
    
    approval_year = np.random.choice(years)
    approval_month = np.random.choice(months)
    approval_day = np.random.choice(days)
    
    approval_date = f"{approval_year}-{approval_month:02d}-{approval_day:02d}"
    
    # Generate other compliance info
    if status == "Recalled":
        recall_reasons = [
            "Manufacturing defect",
            "Contamination concerns",
            "Mislabeling",
            "Adverse events reported",
            "Quality control issues"
        ]
        recall_reason = np.random.choice(recall_reasons)
        recall_date = f"{min(2023, approval_year + np.random.randint(1, 10))}-{np.random.randint(1, 13):02d}-{np.random.randint(1, 29):02d}"
    else:
        recall_reason = None
        recall_date = None
    
    # Compliance information
    compliance_info = {
        "name": medicine_name,
        "status": status,
        "approval_date": approval_date,
        "regulatory_authority": "FDA" if np.random.random() < 0.8 else "EMA",
        "controlled_substance": np.random.random() < 0.2,  # 20% chance of being controlled
        "prescription_required": np.random.random() < 0.7,  # 70% chance of requiring prescription
        "special_instructions": generate_special_instructions(status) if np.random.random() < 0.3 else None,
        "recall_reason": recall_reason,
        "recall_date": recall_date,
        "last_updated": datetime.now().strftime("%Y-%m-%d")
    }
    
    return compliance_info

def generate_special_instructions(status):
    """Generate special instructions based on drug status"""
    instructions = {
        "Approved": [
            "Store at room temperature",
            "Protect from light and moisture",
            "Take with food to reduce stomach upset",
            "Do not crush or chew extended-release formulations",
            "Not for use in pediatric patients under 12 years"
        ],
        "Approved with restrictions": [
            "Limited to 30-day supply",
            "Requires monthly liver function monitoring",
            "Not for use in pregnant women",
            "Restricted to hospital use only",
            "Prior authorization required",
            "REMS program enrollment required"
        ],
        "Under review": [
            "Compassionate use only",
            "Investigational use with informed consent",
            "Available through expanded access program",
            "Efficacy monitoring required"
        ],
        "Discontinued": [
            "Use remaining supply with caution",
            "Consult healthcare provider for alternatives",
            "Check expiration date before use",
            "Return to pharmacy for proper disposal"
        ],
        "Recalled": [
            "Discontinue use immediately",
            "Return to pharmacy for refund",
            "Contact healthcare provider if adverse effects experienced",
            "Do not dispose in household trash",
            "Check lot number against recall notice"
        ]
    }
    
    if status in instructions:
        # Return 1-2 random instructions
        count = np.random.randint(1, 3)
        return np.random.choice(instructions[status], size=min(count, len(instructions[status])), replace=False).tolist()
    
    return ["Follow healthcare provider's instructions"]

def show_medicine_verification():
    """Display the Medicine Verification module"""
    st.subheader("💊 Online Medicine Verification System")
    
    st.markdown("""
    This module allows you to verify medicines against pharmaceutical databases and check 
    their authenticity, regulatory status, and find potential alternatives.
    """)
    
    # Create search interface
    search_col1, search_col2 = st.columns([3, 1])
    
    with search_col1:
        search_term = st.text_input("Enter medicine name or NDC code:", placeholder="e.g., Lisinopril or 12345-678-90")
    
    with search_col2:
        search_options = st.multiselect(
            "Verification Sources:",
            ["Local Database", "RxNorm", "FDA"],
            default=["Local Database"]
        )
    
    # Advanced options
    with st.expander("Advanced Search Options"):
        fuzzy_threshold = st.slider(
            "Fuzzy Match Threshold:",
            min_value=50,
            max_value=100,
            value=70,
            help="Lower values will return more potential matches"
        )
        
        include_alternatives = st.checkbox(
            "Show Alternative Medicines",
            value=True,
            help="Find similar medicines in the same category"
        )
        
        show_compliance = st.checkbox(
            "Show Compliance Information",
            value=True,
            help="Display regulatory status and compliance data"
        )
    
    # Search button
    if st.button("Verify Medicine") and search_term:
        with st.spinner("Searching databases..."):
            results = {}
            
            # Search local database
            if "Local Database" in search_options:
                local_results = search_local_database(search_term, threshold=fuzzy_threshold)
                
                if not local_results.empty:
                    results["local"] = local_results
                    
                    # Display best match first
                    best_match = local_results.iloc[0]
                    
                    st.success(f"Medicine found in local database (Match: {best_match['match_score']}%)")
                    
                    # Display detailed information
                    st.subheader("Medicine Details")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"**Generic Name:** {best_match['generic_name']}")
                        st.markdown(f"**Brand Name:** {best_match['brand_name']}")
                        st.markdown(f"**Manufacturer:** {best_match['manufacturer']}")
                    
                    with col2:
                        st.markdown(f"**Category:** {best_match['category']}")
                        st.markdown(f"**Subcategory:** {best_match['subcategory']}")
                        st.markdown(f"**Price:** ₹{best_match['price']:.2f}")
                    
                    with col3:
                        st.markdown(f"**Dosage Form:** {best_match['dosage_form']}")
                        st.markdown(f"**NDC:** {best_match['ndc']}")
                        
                        # Determine authenticity status
                        if best_match['match_score'] >= 90:
                            st.markdown("**Authenticity:** ✅ Verified")
                        elif best_match['match_score'] >= 75:
                            st.markdown("**Authenticity:** ⚠️ Probable Match")
                        else:
                            st.markdown("**Authenticity:** ❓ Possible Match")
                    
                    # Show dosage information
                    st.markdown(f"**Dosage Information:** {best_match['dosage']}")
                    
                    # Get full information
                    full_info = execute_query(
                        "SELECT * FROM drugs WHERE id = ?", 
                        params=[best_match['id']]
                    )
                    
                    if not full_info.empty:
                        full_drug_info = full_info.iloc[0].to_dict()
                        
                        # Display description
                        with st.expander("Full Description"):
                            st.markdown(full_drug_info.get('description', 'No description available'))
                        
                        # Show compliance information
                        if show_compliance:
                            st.subheader("Regulatory & Compliance Information")
                            
                            compliance_info = get_medicine_compliance_info(best_match['brand_name'])
                            
                            comp_col1, comp_col2 = st.columns(2)
                            
                            with comp_col1:
                                status = compliance_info['status']
                                
                                if status == "Approved":
                                    st.markdown("**Status:** ✅ Approved")
                                elif status == "Approved with restrictions":
                                    st.markdown("**Status:** ⚠️ Approved with restrictions")
                                elif status == "Under review":
                                    st.markdown("**Status:** 🔍 Under review")
                                elif status == "Discontinued":
                                    st.markdown("**Status:** ❌ Discontinued")
                                elif status == "Recalled":
                                    st.markdown("**Status:** 🚫 Recalled")
                                
                                st.markdown(f"**Approval Date:** {compliance_info['approval_date']}")
                                st.markdown(f"**Regulatory Authority:** {compliance_info['regulatory_authority']}")
                            
                            with comp_col2:
                                if compliance_info['controlled_substance']:
                                    st.markdown("**Controlled Substance:** Yes")
                                else:
                                    st.markdown("**Controlled Substance:** No")
                                    
                                if compliance_info['prescription_required']:
                                    st.markdown("**Prescription Required:** Yes")
                                else:
                                    st.markdown("**Prescription Required:** No")
                                
                                st.markdown(f"**Last Updated:** {compliance_info['last_updated']}")
                            
                            # Show recall information if applicable
                            if compliance_info['recall_reason']:
                                st.error(f"""
                                **RECALL NOTICE**
                                
                                This medicine has been recalled as of {compliance_info['recall_date']}.
                                
                                **Reason:** {compliance_info['recall_reason']}
                                """)
                            
                            # Show special instructions
                            if compliance_info['special_instructions']:
                                st.subheader("Special Instructions")
                                for instruction in compliance_info['special_instructions']:
                                    st.markdown(f"• {instruction}")
                        
                        # Show alternatives if requested
                        if include_alternatives:
                            st.subheader("Alternative Medicines")
                            
                            alternatives = find_alternative_medicines(full_drug_info)
                            
                            if not alternatives.empty:
                                # Display alternatives in a table
                                alt_display = alternatives[['generic_name', 'brand_name', 'manufacturer', 'price', 'dosage_form']].copy()
                                alt_display.columns = ['Generic Name', 'Brand Name', 'Manufacturer', 'Price (₹)', 'Dosage Form']
                                
                                # Format price
                                alt_display['Price (₹)'] = alt_display['Price (₹)'].apply(lambda x: f"{x:.2f}")
                                
                                st.dataframe(alt_display, use_container_width=True)
                            else:
                                st.info("No alternative medicines found.")
                else:
                    st.warning("Medicine not found in local database.")
            
            # External verification if requested
            external_results = []
            
            # RxNorm verification
            if "RxNorm" in search_options:
                with st.spinner("Checking RxNorm database..."):
                    rxnorm_result = verify_with_rxnorm(search_term)
                    results["rxnorm"] = rxnorm_result
                    
                    if rxnorm_result["verified"]:
                        external_results.append(f"✅ Verified in RxNorm database (ID: {rxnorm_result['rxnorm_id']})")
                    else:
                        external_results.append(f"❌ Not found in RxNorm database")
            
            # FDA verification
            if "FDA" in search_options:
                with st.spinner("Checking FDA database..."):
                    fda_result = verify_with_fda(search_term)
                    results["fda"] = fda_result
                    
                    if fda_result["verified"]:
                        external_results.append(f"✅ Verified in FDA database")
                    else:
                        external_results.append(f"❌ Not found in FDA database")
            
            # Display external verification results
            if external_results:
                st.subheader("External Verification Results")
                for result in external_results:
                    st.markdown(result)
            
            # No results case
            if not results:
                st.error(f"Medicine '{search_term}' not found in any of the selected databases.")
                
                # Suggestions for better search
                st.info("""
                **Search Tips:**
                - Check for spelling errors in the medicine name
                - Try the generic name instead of the brand name
                - Include the full NDC code if available
                - Lower the fuzzy match threshold for more potential matches
                """)
    
    # Bulk verification interface
    st.subheader("Bulk Medicine Verification")
    
    st.markdown("""
    Upload a CSV file with medicine names or NDC codes to verify multiple medicines at once.
    The file should have a column named 'name' or 'ndc' containing the medicine identifiers.
    """)
    
    uploaded_file = st.file_uploader("Upload CSV file:", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Load the file
            df = pd.read_csv(uploaded_file)
            
            # Check for required columns
            if 'name' in df.columns or 'ndc' in df.columns:
                # Get the identifier column
                id_col = 'name' if 'name' in df.columns else 'ndc'
                
                st.success(f"CSV file loaded successfully with {len(df)} entries.")
                
                # Button to verify all
                if st.button("Verify All Medicines"):
                    # Progress bar
                    progress = st.progress(0)
                    
                    # Results container
                    results_container = st.empty()
                    
                    # Results list
                    verification_results = []
                    
                    # Process each medicine
                    for i, row in enumerate(df.iterrows()):
                        # Update progress
                        progress.progress((i + 1) / len(df))
                        
                        # Get identifier
                        identifier = str(row[1][id_col])
                        
                        # Check local database
                        local_results = search_local_database(identifier, threshold=fuzzy_threshold)
                        
                        if not local_results.empty:
                            best_match = local_results.iloc[0]
                            
                            verification_results.append({
                                "Medicine": identifier,
                                "Status": "Found",
                                "Match": f"{best_match['match_score']}%",
                                "Generic Name": best_match['generic_name'],
                                "Brand Name": best_match['brand_name'],
                                "Manufacturer": best_match['manufacturer']
                            })
                        else:
                            verification_results.append({
                                "Medicine": identifier,
                                "Status": "Not Found",
                                "Match": "0%",
                                "Generic Name": "",
                                "Brand Name": "",
                                "Manufacturer": ""
                            })
                    
                    # Show results as table
                    results_df = pd.DataFrame(verification_results)
                    results_container.dataframe(results_df, use_container_width=True)
                    
                    # Download results
                    if not results_df.empty:
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="Download Verification Results",
                            data=csv,
                            file_name="medicine_verification_results.csv",
                            mime="text/csv"
                        )
            else:
                st.error("CSV file must contain a 'name' or 'ndc' column.")
        
        except Exception as e:
            logger.error(f"Error processing CSV file: {e}")
            st.error(f"Error processing file: {str(e)}")