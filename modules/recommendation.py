"""
Medicine Recommendation module for the Pharmacy Data Science Platform.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3
from config import logger
from db.sqlite_db import execute_query

def create_combined_features(row):
    """Create combined features for recommendation engine"""
    return f"{row['name']} {row['category']} {row['description']} {row['manufacturer']}"

def get_recommendations(drug_id, content_matrix, indices, drugs_df, n=5):
    """
    Get drug recommendations based on content similarity
    
    Args:
        drug_id: ID of the drug to get recommendations for
        content_matrix: Cosine similarity matrix
        indices: Mapping from drug IDs to matrix indices
        drugs_df: DataFrame of drugs
        n: Number of recommendations to return
        
    Returns:
        DataFrame of recommended drugs
    """
    try:
        # Get the index of the drug
        idx = indices[drug_id]
        
        # Get similarity scores
        sim_scores = list(enumerate(content_matrix[idx]))
        
        # Sort based on similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top N similar drugs (excluding the drug itself)
        sim_scores = sim_scores[1:n+1]
        
        # Get drug indices
        drug_indices = [i[0] for i in sim_scores]
        
        # Get similarity values
        similarity_values = [i[1] for i in sim_scores]
        
        # Create recommendation DataFrame
        recommendations = drugs_df.iloc[drug_indices].copy()
        recommendations['similarity'] = similarity_values
        
        return recommendations
    
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        return pd.DataFrame()

def show_medicine_recommendation():
    """Display the Medicine Recommendation module"""
    st.subheader("💊 Medicine Recommendation Engine")
    st.markdown("""
    This module uses Natural Language Processing to recommend similar medicines based on 
    drug characteristics, composition, and therapeutic use.
    """)
    
    try:
        # Load drug data from database
        drugs_df = execute_query("""
            SELECT 
                id, 
                name, 
                category, 
                price, 
                dosage, 
                description, 
                manufacturer, 
                stock
            FROM drugs
            ORDER BY name
        """)
        
        if drugs_df.empty:
            st.warning("No drug data available in the database.")
            return
        
        # Create tabs for different recommendation approaches
        tab1, tab2, tab3 = st.tabs(["Drug Similarity", "Patient Profile", "Sales Pattern"])
        
        # ----- Drug Similarity Recommendations -----
        with tab1:
            st.subheader("Find Similar Drugs")
            
            # Create combined features for content-based filtering
            drugs_df['combined_features'] = drugs_df.apply(create_combined_features, axis=1)
            
            # Create TF-IDF vectorizer
            vectorizer = TfidfVectorizer(stop_words='english')
            
            # Generate TF-IDF matrix
            tfidf_matrix = vectorizer.fit_transform(drugs_df['combined_features'])
            
            # Compute cosine similarity
            cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
            
            # Create a mapping from drug ID to matrix index
            indices = {drugs_df.iloc[i]['id']: i for i in range(len(drugs_df))}
            
            # Select a drug for similarity search
            selected_drug = st.selectbox(
                "Select a drug to find similar alternatives:",
                options=drugs_df['name'].tolist()
            )
            
            # Get the ID of the selected drug
            selected_drug_id = drugs_df[drugs_df['name'] == selected_drug]['id'].iloc[0]
            
            # Get similar drugs
            if st.button("Find Similar Drugs"):
                similar_drugs = get_recommendations(
                    selected_drug_id, cosine_sim, indices, drugs_df, n=5
                )
                
                if not similar_drugs.empty:
                    # Display original drug details
                    st.subheader(f"Selected Drug: {selected_drug}")
                    selected_drug_details = drugs_df[drugs_df['id'] == selected_drug_id].iloc[0]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Category:** {selected_drug_details['category']}")
                        st.markdown(f"**Price:** ₹{selected_drug_details['price']:.2f}")
                        st.markdown(f"**Dosage:** {selected_drug_details['dosage']}")
                    with col2:
                        st.markdown(f"**Manufacturer:** {selected_drug_details['manufacturer']}")
                        st.markdown(f"**Stock Level:** {selected_drug_details['stock']}")
                    
                    st.markdown(f"**Description:** {selected_drug_details['description']}")
                    
                    # Display similar drugs
                    st.subheader("Similar Drugs")
                    
                    # Format the similarity score as percentage
                    similar_drugs['similarity'] = (similar_drugs['similarity'] * 100).round(1).astype(str) + '%'
                    
                    # Display columns of interest
                    display_cols = ['name', 'category', 'price', 'manufacturer', 'similarity']
                    similar_drugs_display = similar_drugs[display_cols].copy()
                    similar_drugs_display.columns = ['Name', 'Category', 'Price (₹)', 'Manufacturer', 'Similarity']
                    
                    st.dataframe(similar_drugs_display, use_container_width=True)
                    
                    # Visualize price comparison
                    st.subheader("Price Comparison")
                    
                    # Create comparison data
                    comparison_data = pd.DataFrame({
                        'Drug': [selected_drug] + similar_drugs['name'].tolist(),
                        'Price': [selected_drug_details['price']] + similar_drugs['price'].tolist(),
                        'Type': ['Selected'] + ['Similar'] * len(similar_drugs)
                    })
                    
                    # Create bar chart
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = ax.bar(comparison_data['Drug'], comparison_data['Price'])
                    
                    # Color the selected drug differently
                    bars[0].set_color('green')
                    
                    # Format chart
                    plt.xticks(rotation=45, ha='right')
                    plt.xlabel('Drug')
                    plt.ylabel('Price (₹)')
                    plt.title('Price Comparison: Selected Drug vs. Similar Alternatives')
                    
                    # Add price labels
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                               f'₹{height:.2f}', ha='center', va='bottom')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.warning("Could not find similar drugs.")
        
        # ----- Patient Profile Recommendations -----
        with tab2:
            st.subheader("Patient Profile-Based Recommendations")
            st.markdown("""
            This section recommends medicines based on patient characteristics and medical conditions.
            Enter patient details to receive personalized recommendations.
            """)
            
            # Patient profile form
            with st.form("patient_profile_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    age = st.number_input("Patient Age:", min_value=0, max_value=120, value=45)
                    gender = st.radio("Gender:", ["Male", "Female", "Other"])
                
                with col2:
                    weight = st.number_input("Weight (kg):", min_value=0, max_value=250, value=70)
                    height = st.number_input("Height (cm):", min_value=0, max_value=250, value=170)
                
                # Medical conditions (multi-select)
                conditions = st.multiselect(
                    "Medical Conditions:",
                    options=[
                        "Hypertension", "Diabetes", "Asthma", "Arthritis", 
                        "Cholesterol", "Heart Disease", "Migraine", "Allergies",
                        "Thyroid Disorder", "Depression", "Anxiety", "Acid Reflux"
                    ]
                )
                
                # Contraindications
                allergies = st.text_area("Allergies or Contraindications (comma separated):", "")
                
                # Current medications
                current_meds = st.text_area("Current Medications (comma separated):", "")
                
                # Submit button
                submit_button = st.form_submit_button("Generate Recommendations")
            
            # Process form submission
            if submit_button:
                # Calculate BMI
                bmi = weight / ((height / 100) ** 2)
                bmi_category = ""
                if bmi < 18.5:
                    bmi_category = "Underweight"
                elif bmi < 25:
                    bmi_category = "Normal weight"
                elif bmi < 30:
                    bmi_category = "Overweight"
                else:
                    bmi_category = "Obese"
                
                # Display patient profile summary
                st.subheader("Patient Profile Summary")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Age", f"{age} years")
                    st.metric("Gender", gender)
                with col2:
                    st.metric("Weight", f"{weight} kg")
                    st.metric("Height", f"{height} cm")
                with col3:
                    st.metric("BMI", f"{bmi:.1f}")
                    st.metric("BMI Category", bmi_category)
                
                st.subheader("Medical Conditions")
                if conditions:
                    st.write(", ".join(conditions))
                else:
                    st.write("None specified")
                
                # Map conditions to drug categories (simplified for demonstration)
                condition_to_category = {
                    "Hypertension": "Cardiovascular",
                    "Heart Disease": "Cardiovascular",
                    "Cholesterol": "Cardiovascular",
                    "Diabetes": "Antidiabetic",
                    "Asthma": "Respiratory",
                    "Allergies": "Respiratory",
                    "Migraine": "Pain Management",
                    "Arthritis": "Pain Management",
                    "Depression": "Psychiatric",
                    "Anxiety": "Psychiatric",
                    "Acid Reflux": "Gastrointestinal",
                    "Thyroid Disorder": "Hormonal"
                }
                
                # Get recommended categories based on conditions
                recommended_categories = [condition_to_category.get(condition) for condition in conditions if condition in condition_to_category]
                recommended_categories = list(set(recommended_categories))  # Remove duplicates
                
                # Parse allergies
                allergy_list = [a.strip() for a in allergies.split(",") if a.strip()]
                
                # Generate recommendations
                st.subheader("Recommended Medications")
                
                if not recommended_categories:
                    st.info("No specific drug categories identified based on the provided medical conditions.")
                else:
                    for category in recommended_categories:
                        st.markdown(f"**For {category} conditions:**")
                        
                        # Get drugs in this category
                        category_drugs = drugs_df[drugs_df['category'] == category].copy()
                        
                        if category_drugs.empty:
                            st.write("No medications available in this category.")
                        else:
                            # Filter out allergies (simplified)
                            if allergy_list:
                                for allergy in allergy_list:
                                    category_drugs = category_drugs[
                                        ~category_drugs['name'].str.contains(allergy, case=False) &
                                        ~category_drugs['description'].str.contains(allergy, case=False)
                                    ]
                            
                            # Sort by price for options at different price points
                            category_drugs = category_drugs.sort_values('price')
                            
                            if category_drugs.empty:
                                st.write("No suitable medications found (potential allergies filtered out).")
                            else:
                                # Display options
                                display_cols = ['name', 'dosage', 'price', 'manufacturer']
                                display_df = category_drugs[display_cols].head(3)
                                display_df.columns = ['Name', 'Dosage', 'Price (₹)', 'Manufacturer']
                                st.dataframe(display_df)
                
                # Considerations based on age and conditions
                st.subheader("Clinical Considerations")
                
                considerations = []
                
                # Age-based considerations
                if age >= 65:
                    considerations.append("• Elderly patients may require lower dosages due to reduced kidney function")
                    considerations.append("• Monitor for increased risk of side effects and drug interactions")
                
                if age <= 12:
                    considerations.append("• Pediatric dosing required - verify all calculations")
                    considerations.append("• Many adult formulations may not be appropriate for children")
                
                # BMI-based considerations
                if bmi < 18.5 or bmi >= 30:
                    considerations.append("• Dosage adjustments may be needed based on BMI")
                
                # Condition-based considerations
                if "Diabetes" in conditions and "Hypertension" in conditions:
                    considerations.append("• Monitor for drug interactions between antidiabetic and cardiovascular medications")
                
                if "Kidney Disease" in conditions:
                    considerations.append("• Renal dosing adjustments required for many medications")
                
                if considerations:
                    for c in considerations:
                        st.markdown(c)
                else:
                    st.write("No special clinical considerations identified.")
        
        # ----- Sales Pattern Recommendations -----
        with tab3:
            st.subheader("Sales Pattern Analysis")
            st.markdown("""
            This section analyzes shopping patterns and makes recommendations based on 
            frequently co-purchased items.
            """)
            
            # Get sales pattern data (simplified for demonstration)
            try:
                # This would normally use association rule mining
                # We'll simulate co-purchase patterns for demonstration
                
                # Get some sample combinations
                co_purchase_patterns = [
                    {"main_drug": "Drug-001", "category": "Cardiovascular", "frequently_bought_with": ["Drug-010", "Drug-015"]},
                    {"main_drug": "Drug-010", "category": "Antidiabetic", "frequently_bought_with": ["Drug-001", "Drug-025"]},
                    {"main_drug": "Drug-028", "category": "Respiratory", "frequently_bought_with": ["Drug-030", "Drug-032"]},
                    {"main_drug": "Drug-045", "category": "Pain Management", "frequently_bought_with": ["Drug-048", "Drug-050"]},
                    {"main_drug": "Drug-060", "category": "Gastrointestinal", "frequently_bought_with": ["Drug-062", "Drug-064"]}
                ]
                
                # Convert to DataFrame for easier handling
                patterns_df = pd.DataFrame(co_purchase_patterns)
                
                # Add detailed drug info
                main_drugs = []
                for drug_name in patterns_df['main_drug']:
                    drug_details = drugs_df[drugs_df['name'] == drug_name]
                    if not drug_details.empty:
                        main_drugs.append(drug_details.iloc[0])
                    else:
                        main_drugs.append(None)
                
                valid_patterns = [i for i, d in enumerate(main_drugs) if d is not None]
                patterns_df = patterns_df.iloc[valid_patterns].reset_index(drop=True)
                main_drugs = [d for d in main_drugs if d is not None]
                
                if not patterns_df.empty:
                    # Select drug to get recommendations for
                    drug_options = [d['name'] for d in main_drugs]
                    selected_main_drug = st.selectbox("Select a drug to see co-purchase recommendations:", drug_options)
                    
                    # Get index of selected drug
                    selected_idx = drug_options.index(selected_main_drug)
                    
                    # Get co-purchased drugs
                    co_purchased = patterns_df.iloc[selected_idx]['frequently_bought_with']
                    
                    # Display co-purchase information
                    st.subheader(f"Customers who purchased {selected_main_drug} also bought:")
                    
                    # Create a table of co-purchased drugs
                    co_purchase_details = []
                    for drug_name in co_purchased:
                        drug_info = drugs_df[drugs_df['name'] == drug_name]
                        if not drug_info.empty:
                            co_purchase_details.append(drug_info.iloc[0])
                    
                    if co_purchase_details:
                        co_df = pd.DataFrame(co_purchase_details)
                        display_cols = ['name', 'category', 'price', 'manufacturer']
                        co_display = co_df[display_cols].copy()
                        co_display.columns = ['Name', 'Category', 'Price (₹)', 'Manufacturer']
                        st.dataframe(co_display)
                        
                        # Bundle recommendation
                        st.subheader("Bundle Offer")
                        
                        # Calculate bundle details
                        selected_drug_info = drugs_df[drugs_df['name'] == selected_main_drug].iloc[0]
                        bundle_items = [selected_drug_info] + co_purchase_details
                        
                        # Create a table showing the bundle
                        bundle_df = pd.DataFrame({
                            'Product': [item['name'] for item in bundle_items],
                            'Regular Price': [item['price'] for item in bundle_items],
                        })
                        
                        # Calculate total and discounted price
                        regular_total = bundle_df['Regular Price'].sum()
                        bundle_discount = 0.1  # 10% discount
                        discounted_total = regular_total * (1 - bundle_discount)
                        savings = regular_total - discounted_total
                        
                        # Add bundle pricing
                        bundle_df['Bundle Price'] = bundle_df['Regular Price'] * (1 - bundle_discount)
                        bundle_df['Savings'] = bundle_df['Regular Price'] - bundle_df['Bundle Price']
                        
                        # Format currencies
                        for col in ['Regular Price', 'Bundle Price', 'Savings']:
                            bundle_df[col] = bundle_df[col].apply(lambda x: f"₹{x:.2f}")
                        
                        # Display bundle details
                        st.dataframe(bundle_df)
                        
                        # Summary
                        st.markdown(f"""
                        **Bundle Summary:**
                        - Regular Total: ₹{regular_total:.2f}
                        - Bundle Price: ₹{discounted_total:.2f}
                        - You Save: ₹{savings:.2f} ({bundle_discount*100:.0f}%)
                        """)
                    else:
                        st.info("No co-purchase data available for this product.")
                else:
                    st.info("No sales pattern data available for analysis.")
            
            except Exception as e:
                logger.error(f"Error in sales pattern analysis: {e}")
                st.error(f"Error analyzing sales patterns: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error in medicine recommendation: {e}")
        st.error(f"Error loading recommendation engine: {str(e)}")