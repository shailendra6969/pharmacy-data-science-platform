"""
Gene-Drug Analysis module for the Pharmacy Data Science Platform.
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests
import time
from config import logger, API_TIMEOUT, API_RETRY_ATTEMPTS
from utils.data_loader import load_uploaded_data

def fetch_gene_data(gene_ids):
    """
    Fetch gene-drug interaction data from PharmGKB API with retry logic
    
    Args:
        gene_ids: List of PharmGKB gene IDs
    
    Returns:
        DataFrame containing gene-drug interactions
    """
    gene_drug_data = []
    
    for gene_id in gene_ids:
        gene_id = gene_id.strip()
        if not gene_id:
            continue
            
        for attempt in range(API_RETRY_ATTEMPTS):
            try:
                st.write(f"Fetching data for gene ID: {gene_id} (Attempt {attempt+1}/{API_RETRY_ATTEMPTS})")
                url = f"https://api.pharmgkb.org/v1/data/gene/{gene_id}"
                response = requests.get(url, timeout=API_TIMEOUT)
                
                if response.status_code == 200:
                    data = response.json()
                    gene_name = data.get('name', 'Unknown')
                    related_drugs = data.get('relatedChemicals', [])
                    
                    if not related_drugs:
                        st.info(f"No drug interactions found for gene {gene_id} ({gene_name})")
                    
                    for drug in related_drugs:
                        gene_drug_data.append({
                            "Gene": gene_name, 
                            "Gene ID": gene_id,
                            "Drug": drug.get('name', 'Unknown Drug'),
                            "Relation": drug.get('relation', 'Unknown')
                        })
                    
                    # Break the retry loop on success
                    break
                    
                elif response.status_code == 404:
                    st.warning(f"Gene ID {gene_id} not found. Verify the ID is correct.")
                    break
                    
                else:
                    st.warning(f"API error for gene {gene_id}. Status code: {response.status_code}")
                    if attempt < API_RETRY_ATTEMPTS - 1:
                        time.sleep(1 * (attempt + 1))  # Exponential backoff
                        continue
                    else:
                        st.error(f"Failed to fetch data for gene {gene_id} after {API_RETRY_ATTEMPTS} attempts")
                
            except requests.exceptions.Timeout:
                st.warning(f"Request timed out for gene {gene_id}")
                if attempt < API_RETRY_ATTEMPTS - 1:
                    time.sleep(1 * (attempt + 1))
                    continue
                else:
                    st.error(f"Request timed out for gene {gene_id} after {API_RETRY_ATTEMPTS} attempts")
                    
            except Exception as e:
                logger.error(f"Error fetching data for gene {gene_id}: {e}")
                st.error(f"Error: {str(e)}")
                break
                
            # Wait between requests to avoid API rate limits
            time.sleep(0.5)
    
    return pd.DataFrame(gene_drug_data)

def show_gene_analysis():
    """Display the Gene-Drug Analysis module"""
    st.subheader("🧬 PharmGKB Gene–Drug Interaction Analysis")
    st.info("""
    This module analyzes gene-drug interactions using the PharmGKB API. You can:
    1. Upload a CSV file containing gene IDs and drug names
    2. Enter PharmGKB gene IDs directly
    3. Visualize gene-drug interaction networks
    """)
    
    tab1, tab2 = st.tabs(["Data Input", "Network Analysis"])
    
    with tab1:
        uploaded_file = st.file_uploader("Upload CSV with Gene IDs:", type=["csv"])
        gene_ids_input = st.text_area("Or Enter PharmGKB Gene IDs (comma separated):", 
                                    "PA124,PA128,PA130,PA131,PA134,PA151")
        
        if uploaded_file is not None:
            df = load_uploaded_data(uploaded_file)
            if df is not None:
                st.success("CSV Uploaded Successfully")
                if 'Gene ID' in df.columns:
                    gene_id_list = df['Gene ID'].unique().tolist()
                    st.info(f"Found {len(gene_id_list)} unique gene IDs in the CSV file")
                    if st.button("Fetch Gene Data from CSV"):
                        with st.spinner("Fetching gene-drug interaction data..."):
                            gene_data = fetch_gene_data(gene_id_list)
                            if not gene_data.empty:
                                st.dataframe(gene_data)
                                st.session_state.gene_data = gene_data
                else:
                    st.warning("The CSV file must contain a 'Gene ID' column")
        
        elif st.button("Fetch Gene–Drug Data"):
            gene_id_list = [gid.strip() for gid in gene_ids_input.split(",") if gid.strip()]
            if not gene_id_list:
                st.error("Please enter at least one gene ID")
            else:
                with st.spinner("Fetching gene-drug interaction data..."):
                    gene_data = fetch_gene_data(gene_id_list)
                    if not gene_data.empty:
                        st.dataframe(gene_data)
                        st.session_state.gene_data = gene_data
                        
                        # Show simple bar chart of interactions
                        fig, ax = plt.subplots(figsize=(10, 6))
                        gene_counts = gene_data['Gene'].value_counts()
                        gene_counts.plot(kind='bar', ax=ax)
                        ax.set_title('Number of Drug Interactions by Gene')
                        ax.set_xlabel('Gene')
                        ax.set_ylabel('Number of Interactions')
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.warning("No data found for the provided gene IDs")
    
    with tab2:
        st.subheader("Gene-Drug Interaction Network")
        
        if 'gene_data' not in st.session_state:
            st.info("Please fetch gene-drug data first in the 'Data Input' tab")
        elif st.session_state.gene_data.empty:
            st.info("No gene-drug interaction data available. Please fetch data in the 'Data Input' tab.")
        else:
            # Import here to avoid loading networkx if not needed
            import networkx as nx
            
            try:
                df = st.session_state.gene_data
                
                # Create graph
                G = nx.Graph()
                
                # Add gene nodes
                genes = df['Gene'].unique()
                for gene in genes:
                    G.add_node(gene, type='gene')
                
                # Add drug nodes
                drugs = df['Drug'].unique()
                for drug in drugs:
                    G.add_node(drug, type='drug')
                
                # Add edges
                for _, row in df.iterrows():
                    G.add_edge(row['Gene'], row['Drug'], relation=row.get('Relation', 'Unknown'))
                
                # Calculate network metrics
                gene_centrality = nx.degree_centrality(G)
                betweenness = nx.betweenness_centrality(G)
                
                # Visualization options
                st.subheader("Network Visualization Options")
                col1, col2 = st.columns(2)
                with col1:
                    gene_node_size = st.slider("Gene Node Size:", 100, 1500, 800)
                    drug_node_size = st.slider("Drug Node Size:", 100, 1000, 400)
                
                with col2:
                    gene_color = st.color_picker("Gene Node Color:", "#ADD8E6")  # Light blue
                    drug_color = st.color_picker("Drug Node Color:", "#90EE90")  # Light green
                
                # Draw network
                fig, ax = plt.subplots(figsize=(12, 10))
                
                # Position nodes using spring layout
                pos = nx.spring_layout(G, seed=42)
                
                # Draw gene nodes
                gene_nodes = [n for n in G.nodes() if n in genes]
                nx.draw_networkx_nodes(G, pos, nodelist=gene_nodes, node_size=gene_node_size, 
                                      node_color=gene_color, alpha=0.8, label='Genes')
                
                # Draw drug nodes
                drug_nodes = [n for n in G.nodes() if n in drugs]
                nx.draw_networkx_nodes(G, pos, nodelist=drug_nodes, node_size=drug_node_size,
                                      node_color=drug_color, alpha=0.6, label='Drugs')
                
                # Draw edges
                nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
                
                # Draw labels
                gene_labels = {n: n for n in gene_nodes}
                drug_labels = {n: n for n in drug_nodes}
                nx.draw_networkx_labels(G, pos, labels=gene_labels, font_size=12, font_weight='bold')
                nx.draw_networkx_labels(G, pos, labels=drug_labels, font_size=8)
                
                plt.title("Gene-Drug Interaction Network", fontsize=16)
                plt.legend(scatterpoints=1)
                plt.axis('off')
                st.pyplot(fig)
                
                # Network Metrics Analysis
                st.subheader("Network Centrality Analysis")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Degree Centrality (Top 5)")
                    centrality_df = pd.DataFrame({
                        'Node': list(gene_centrality.keys()),
                        'Centrality': list(gene_centrality.values())
                    }).sort_values('Centrality', ascending=False).head(5)
                    st.dataframe(centrality_df)
                    
                    if not centrality_df.empty:
                        st.markdown(f"""
                        **Key Insight**: {centrality_df.iloc[0]['Node']} is the most connected node, 
                        interacting with {int(centrality_df.iloc[0]['Centrality'] * (len(G.nodes) - 1))} other nodes.
                        This suggests it may be a hub gene involved in multiple drug responses.
                        """)
                
                with col2:
                    st.subheader("Betweenness Centrality (Top 5)")
                    betweenness_df = pd.DataFrame({
                        'Node': list(betweenness.keys()),
                        'Betweenness': list(betweenness.values())
                    }).sort_values('Betweenness', ascending=False).head(5)
                    st.dataframe(betweenness_df)
                    
                    if not betweenness_df.empty:
                        st.markdown(f"""
                        **Key Insight**: {betweenness_df.iloc[0]['Node']} has the highest betweenness centrality,
                        suggesting it may act as a bridge between different drug interaction pathways.
                        """)
                
                # Export options
                st.subheader("Export Options")
                if st.button("Export Gene-Drug Data to CSV"):
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="gene_drug_interactions.csv",
                        mime="text/csv"
                    )
            
            except Exception as e:
                logger.error(f"Error in network analysis: {e}")
                st.error(f"Error generating network visualization: {str(e)}")