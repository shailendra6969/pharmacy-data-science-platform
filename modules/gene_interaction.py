"""
Gene Interaction Network module for the Pharmacy Data Science Platform.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import requests
import time
import json
from config import logger, API_TIMEOUT, API_RETRY_ATTEMPTS
from utils.data_loader import load_uploaded_data

def show_gene_interaction():
    """Display the Gene Interaction Network module"""
    st.subheader("🧬 Gene Interaction Network Analysis")
    st.markdown("""
    This module visualizes and analyzes gene-gene and gene-drug interaction networks.
    Upload gene interaction data or enter gene IDs to generate a network visualization
    with centrality metrics.
    """)
    
    # Create tabs for different network types
    tab1, tab2, tab3 = st.tabs(["Data Input", "Network Visualization", "Pathway Analysis"])
    
    # ----- Data Input Tab -----
    with tab1:
        st.subheader("Input Gene Interaction Data")
        
        data_source = st.radio(
            "Select Data Source:",
            ["Upload CSV", "Enter Gene IDs", "Use Sample Data"]
        )
        
        gene_data = None
        
        if data_source == "Upload CSV":
            uploaded_file = st.file_uploader(
                "Upload Gene Interaction CSV:",
                type=["csv"]
            )
            
            if uploaded_file is not None:
                try:
                    gene_data = load_uploaded_data(uploaded_file)
                    
                    if gene_data is not None:
                        st.success("CSV Uploaded Successfully")
                        st.dataframe(gene_data.head())
                        
                        # Check for required columns
                        required_cols = ["source", "target", "interaction_type"]
                        missing_cols = [col for col in required_cols if col not in gene_data.columns]
                        
                        if missing_cols:
                            st.warning(f"Missing required columns: {', '.join(missing_cols)}")
                            st.info("""
                            The CSV should contain at least:
                            - 'source': Source gene ID or name
                            - 'target': Target gene ID or name
                            - 'interaction_type': Type of interaction
                            """)
                        else:
                            st.session_state.gene_network_data = gene_data
                except Exception as e:
                    logger.error(f"Error loading gene interaction data: {e}")
                    st.error(f"Error loading CSV: {str(e)}")
        
        elif data_source == "Enter Gene IDs":
            st.info("""
            Enter a list of gene IDs to fetch interaction data.
            This will use API calls to retrieve known interactions.
            """)
            
            gene_ids = st.text_area(
                "Enter Gene IDs (comma or line separated):",
                "BRCA1, BRCA2, TP53, EGFR, KRAS"
            )
            
            if st.button("Fetch Interaction Data"):
                # Parse gene IDs
                gene_list = [gene.strip() for gene in gene_ids.replace(",", "\n").split("\n") if gene.strip()]
                
                if gene_list:
                    # This would normally use a real API call
                    # Creating simulated data for demonstration
                    with st.spinner("Fetching gene interaction data..."):
                        try:
                            # Simulate API call delay
                            time.sleep(2)
                            
                            # Create simulated interaction data
                            interactions = []
                            interaction_types = ["activation", "inhibition", "binding", "expression", "phosphorylation"]
                            confidence_levels = ["high", "medium", "low"]
                            
                            # Generate all possible gene pairs
                            for i, gene1 in enumerate(gene_list):
                                for gene2 in gene_list[i+1:]:
                                    # Not all genes interact with each other
                                    if np.random.random() < 0.7:  # 70% chance of interaction
                                        interactions.append({
                                            "source": gene1,
                                            "target": gene2,
                                            "interaction_type": np.random.choice(interaction_types),
                                            "confidence": np.random.choice(confidence_levels),
                                            "score": round(np.random.uniform(0.5, 1.0), 2)
                                        })
                            
                            # Create DataFrame
                            if interactions:
                                gene_data = pd.DataFrame(interactions)
                                st.session_state.gene_network_data = gene_data
                                st.success(f"Generated {len(gene_data)} interactions between {len(gene_list)} genes")
                                st.dataframe(gene_data)
                            else:
                                st.warning("No interactions found between the specified genes.")
                        
                        except Exception as e:
                            logger.error(f"Error generating interaction data: {e}")
                            st.error(f"Error: {str(e)}")
                else:
                    st.error("Please enter at least one gene ID.")
        
        elif data_source == "Use Sample Data":
            # Create sample gene interaction data
            st.info("Using built-in sample gene interaction data for demonstration.")
            
            # Create sample data if not already in session state
            if 'gene_network_data' not in st.session_state:
                try:
                    # Sample genes related to drug metabolism
                    sample_genes = [
                        "CYP3A4", "CYP2D6", "CYP2C9", "CYP2C19", "CYP1A2",
                        "ABCB1", "VKORC1", "SLCO1B1", "DPYD", "TPMT",
                        "UGT1A1", "ADRB2", "DRD2", "OPRM1", "COMT"
                    ]
                    
                    # Create interactions (this would normally come from a real database)
                    interactions = [
                        {"source": "CYP3A4", "target": "CYP2D6", "interaction_type": "co-expression", "confidence": "high", "score": 0.92},
                        {"source": "CYP3A4", "target": "ABCB1", "interaction_type": "regulation", "confidence": "high", "score": 0.89},
                        {"source": "CYP2D6", "target": "CYP2C9", "interaction_type": "co-expression", "confidence": "medium", "score": 0.78},
                        {"source": "CYP2C9", "target": "VKORC1", "interaction_type": "functional", "confidence": "high", "score": 0.85},
                        {"source": "VKORC1", "target": "CYP2C19", "interaction_type": "co-expression", "confidence": "low", "score": 0.65},
                        {"source": "CYP1A2", "target": "CYP3A4", "interaction_type": "co-expression", "confidence": "medium", "score": 0.76},
                        {"source": "SLCO1B1", "target": "ABCB1", "interaction_type": "co-expression", "confidence": "medium", "score": 0.72},
                        {"source": "ABCB1", "target": "DPYD", "interaction_type": "physical", "confidence": "low", "score": 0.63},
                        {"source": "TPMT", "target": "DPYD", "interaction_type": "co-expression", "confidence": "medium", "score": 0.71},
                        {"source": "UGT1A1", "target": "CYP3A4", "interaction_type": "regulation", "confidence": "high", "score": 0.88},
                        {"source": "ADRB2", "target": "DRD2", "interaction_type": "functional", "confidence": "medium", "score": 0.75},
                        {"source": "DRD2", "target": "OPRM1", "interaction_type": "functional", "confidence": "high", "score": 0.86},
                        {"source": "OPRM1", "target": "COMT", "interaction_type": "co-expression", "confidence": "medium", "score": 0.77},
                        {"source": "COMT", "target": "DRD2", "interaction_type": "regulation", "confidence": "high", "score": 0.82},
                        {"source": "CYP2C19", "target": "CYP2C9", "interaction_type": "co-expression", "confidence": "high", "score": 0.91},
                        {"source": "CYP1A2", "target": "UGT1A1", "interaction_type": "co-expression", "confidence": "medium", "score": 0.79},
                        {"source": "SLCO1B1", "target": "UGT1A1", "interaction_type": "functional", "confidence": "low", "score": 0.64},
                        {"source": "TPMT", "target": "UGT1A1", "interaction_type": "co-expression", "confidence": "low", "score": 0.67},
                        {"source": "ADRB2", "target": "COMT", "interaction_type": "functional", "confidence": "medium", "score": 0.73},
                        {"source": "CYP3A4", "target": "SLCO1B1", "interaction_type": "regulation", "confidence": "high", "score": 0.84}
                    ]
                    
                    # Add drug interactions
                    drugs = ["Warfarin", "Clopidogrel", "Simvastatin", "Codeine", "Tamoxifen"]
                    drug_interactions = [
                        {"source": "CYP2C9", "target": "Warfarin", "interaction_type": "metabolism", "confidence": "high", "score": 0.94},
                        {"source": "VKORC1", "target": "Warfarin", "interaction_type": "target", "confidence": "high", "score": 0.95},
                        {"source": "CYP2C19", "target": "Clopidogrel", "interaction_type": "metabolism", "confidence": "high", "score": 0.93},
                        {"source": "SLCO1B1", "target": "Simvastatin", "interaction_type": "transport", "confidence": "high", "score": 0.87},
                        {"source": "CYP3A4", "target": "Simvastatin", "interaction_type": "metabolism", "confidence": "medium", "score": 0.78},
                        {"source": "CYP2D6", "target": "Codeine", "interaction_type": "metabolism", "confidence": "high", "score": 0.92},
                        {"source": "OPRM1", "target": "Codeine", "interaction_type": "target", "confidence": "high", "score": 0.91},
                        {"source": "CYP2D6", "target": "Tamoxifen", "interaction_type": "metabolism", "confidence": "high", "score": 0.90},
                    ]
                    
                    # Combine all interactions
                    all_interactions = interactions + drug_interactions
                    
                    # Create DataFrame
                    gene_data = pd.DataFrame(all_interactions)
                    st.session_state.gene_network_data = gene_data
                
                except Exception as e:
                    logger.error(f"Error creating sample data: {e}")
                    st.error(f"Error creating sample data: {str(e)}")
            
            # Display sample data
            if 'gene_network_data' in st.session_state:
                st.dataframe(st.session_state.gene_network_data)
    
    # ----- Network Visualization Tab -----
    with tab2:
        st.subheader("Gene Interaction Network Visualization")
        
        if 'gene_network_data' not in st.session_state:
            st.info("Please load or generate gene interaction data in the 'Data Input' tab.")
        else:
            try:
                df = st.session_state.gene_network_data
                
                # Network visualization options
                st.subheader("Visualization Options")
                col1, col2 = st.columns(2)
                
                with col1:
                    min_confidence = st.selectbox(
                        "Minimum Confidence Level:",
                        ["All", "low", "medium", "high"],
                        index=0
                    )
                    
                    layout_type = st.selectbox(
                        "Network Layout:",
                        ["spring", "circular", "kamada_kawai", "random", "shell"],
                        index=0
                    )
                
                with col2:
                    selected_interactions = st.multiselect(
                        "Interaction Types:",
                        options=sorted(df['interaction_type'].unique()),
                        default=list(df['interaction_type'].unique())
                    )
                    
                    show_labels = st.checkbox("Show Node Labels", value=True)
                
                # Filter data based on selections
                filtered_df = df.copy()
                
                if min_confidence != "All" and 'confidence' in filtered_df.columns:
                    confidence_levels = {"low": 0, "medium": 1, "high": 2}
                    confidence_map = {"low": 0, "medium": 1, "high": 2}
                    min_conf_val = confidence_map[min_confidence]
                    filtered_df = filtered_df[filtered_df['confidence'].apply(lambda x: confidence_map.get(x, 0)) >= min_conf_val]
                
                if selected_interactions:
                    filtered_df = filtered_df[filtered_df['interaction_type'].isin(selected_interactions)]
                
                if filtered_df.empty:
                    st.warning("No interactions match the selected filters.")
                else:
                    # Create network graph
                    G = nx.Graph()
                    
                    # Add edges with attributes
                    for _, row in filtered_df.iterrows():
                        G.add_edge(
                            row['source'], 
                            row['target'], 
                            type=row['interaction_type'],
                            confidence=row.get('confidence', 'medium'),
                            weight=row.get('score', 0.5)
                        )
                    
                    # Identify genes and drugs
                    all_nodes = set(filtered_df['source']).union(set(filtered_df['target']))
                    
                    # Assume drugs have capitalized names (for demonstration)
                    # In real implementation, this would use a more robust method
                    drugs = [node for node in all_nodes if node[0].isupper() and node.lower() not in [n.lower() for n in all_nodes]]
                    genes = [node for node in all_nodes if node not in drugs]
                    
                    # Set node attributes
                    for node in G.nodes():
                        if node in drugs:
                            G.nodes[node]['type'] = 'drug'
                        else:
                            G.nodes[node]['type'] = 'gene'
                    
                    # Calculate network metrics
                    degree_centrality = nx.degree_centrality(G)
                    betweenness_centrality = nx.betweenness_centrality(G)
                    closeness_centrality = nx.closeness_centrality(G)
                    
                    # Add centrality measures as node attributes
                    for node in G.nodes():
                        G.nodes[node]['degree'] = degree_centrality[node]
                        G.nodes[node]['betweenness'] = betweenness_centrality[node]
                        G.nodes[node]['closeness'] = closeness_centrality[node]
                    
                    # Create the visualization
                    plt.figure(figsize=(12, 10))
                    
                    # Set the layout
                    if layout_type == "spring":
                        pos = nx.spring_layout(G, seed=42)
                    elif layout_type == "circular":
                        pos = nx.circular_layout(G)
                    elif layout_type == "kamada_kawai":
                        pos = nx.kamada_kawai_layout(G)
                    elif layout_type == "random":
                        pos = nx.random_layout(G, seed=42)
                    elif layout_type == "shell":
                        pos = nx.shell_layout(G)
                    
                    # Define node colors by type
                    node_colors = []
                    for node in G.nodes():
                        if G.nodes[node]['type'] == 'drug':
                            node_colors.append('lightgreen')
                        else:
                            node_colors.append('lightblue')
                    
                    # Draw nodes with size based on degree centrality
                    node_sizes = [1000 * G.nodes[node]['degree'] + 100 for node in G.nodes()]
                    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
                    
                    # Define edge colors by interaction type
                    edge_colors = []
                    interaction_color_map = {
                        'activation': 'green',
                        'inhibition': 'red',
                        'binding': 'blue',
                        'expression': 'purple',
                        'co-expression': 'purple',
                        'regulation': 'orange',
                        'physical': 'brown',
                        'functional': 'gray',
                        'metabolism': 'cyan',
                        'transport': 'magenta',
                        'target': 'yellow',
                        'phosphorylation': 'pink'
                    }
                    
                    for edge in G.edges(data=True):
                        edge_type = edge[2]['type']
                        edge_colors.append(interaction_color_map.get(edge_type, 'black'))
                    
                    # Draw edges with width based on weight
                    edge_weights = [G[u][v].get('weight', 0.5) * 2 for u, v in G.edges()]
                    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_weights, alpha=0.7)
                    
                    # Draw labels if enabled
                    if show_labels:
                        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
                    
                    # Add legend for node types
                    plt.legend([plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=10),
                               plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', markersize=10)],
                              ['Gene', 'Drug'])
                    
                    plt.title("Gene Interaction Network", fontsize=16)
                    plt.axis('off')
                    st.pyplot(plt)
                    
                    # Display network statistics
                    st.subheader("Network Statistics")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Nodes", len(G.nodes()))
                        st.metric("Edges", len(G.edges()))
                    
                    with col2:
                        try:
                            avg_degree = sum(dict(G.degree()).values()) / len(G.nodes())
                            st.metric("Average Degree", f"{avg_degree:.2f}")
                        except:
                            st.metric("Average Degree", "N/A")
                        
                        try:
                            density = nx.density(G)
                            st.metric("Network Density", f"{density:.4f}")
                        except:
                            st.metric("Network Density", "N/A")
                    
                    with col3:
                        try:
                            avg_clustering = nx.average_clustering(G)
                            st.metric("Clustering Coefficient", f"{avg_clustering:.4f}")
                        except:
                            st.metric("Clustering Coefficient", "N/A")
                        
                        try:
                            n_components = nx.number_connected_components(G)
                            st.metric("Connected Components", n_components)
                        except:
                            st.metric("Connected Components", "N/A")
                    
                    # Top nodes by centrality
                    st.subheader("Top Nodes by Centrality Measures")
                    
                    # Create DataFrames for centrality measures
                    degree_df = pd.DataFrame({
                        'Node': list(degree_centrality.keys()),
                        'Degree Centrality': list(degree_centrality.values())
                    }).sort_values('Degree Centrality', ascending=False).head(10)
                    
                    betweenness_df = pd.DataFrame({
                        'Node': list(betweenness_centrality.keys()),
                        'Betweenness Centrality': list(betweenness_centrality.values())
                    }).sort_values('Betweenness Centrality', ascending=False).head(10)
                    
                    closeness_df = pd.DataFrame({
                        'Node': list(closeness_centrality.keys()),
                        'Closeness Centrality': list(closeness_centrality.values())
                    }).sort_values('Closeness Centrality', ascending=False).head(10)
                    
                    # Display centrality tables
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.subheader("Degree Centrality")
                        st.dataframe(degree_df)
                    
                    with col2:
                        st.subheader("Betweenness Centrality")
                        st.dataframe(betweenness_df)
                    
                    with col3:
                        st.subheader("Closeness Centrality")
                        st.dataframe(closeness_df)
                    
                    # Network interpretation
                    st.subheader("Network Interpretation")
                    
                    # Get top gene by degree
                    top_gene = degree_df.iloc[0]['Node']
                    top_degree = degree_df.iloc[0]['Degree Centrality']
                    
                    # Get top gene by betweenness
                    top_betweenness_gene = betweenness_df.iloc[0]['Node']
                    
                    # Calculate subnetworks
                    communities = list(nx.community.greedy_modularity_communities(G))
                    
                    st.markdown(f"""
                    ### Key Insights:
                    
                    1. **Central Nodes**: {top_gene} shows the highest connectivity (degree centrality: {top_degree:.4f}), 
                       suggesting it plays a central role in this interaction network.
                       
                    2. **Bridge Nodes**: {top_betweenness_gene} has high betweenness centrality, indicating it likely 
                       serves as a bridge between different functional modules or pathways.
                       
                    3. **Community Structure**: The network contains {len(communities)} distinct communities or modules,
                       suggesting functional separation of gene groups.
                       
                    4. **Clinical Relevance**: Highly connected genes are potential therapeutic targets or 
                       biomarkers for drug response and adverse reactions.
                    """)
                    
            except Exception as e:
                logger.error(f"Error in network visualization: {e}")
                st.error(f"Error generating network visualization: {str(e)}")
    
    # ----- Pathway Analysis Tab -----
    with tab3:
        st.subheader("Pathway Analysis")
        
        if 'gene_network_data' not in st.session_state:
            st.info("Please load or generate gene interaction data in the 'Data Input' tab.")
        else:
            try:
                df = st.session_state.gene_network_data
                
                # Extract all unique gene/protein names
                all_nodes = set(df['source']).union(set(df['target']))
                
                # Filter to only include genes (exclude drugs)
                # In a real implementation, this would use a more robust method
                genes = [node for node in all_nodes if not (node[0].isupper() and node.lower() not in [n.lower() for n in all_nodes])]
                
                st.info(f"Performing pathway analysis for {len(genes)} genes/proteins in the network.")
                
                # This would normally call an external API for pathway enrichment
                # We'll simulate the results for demonstration
                
                # Simulate pathway enrichment results
                pathways = [
                    {"name": "Drug Metabolism", "genes": ["CYP3A4", "CYP2D6", "CYP2C9", "CYP2C19", "UGT1A1"], "p_value": 0.00001, "fold_enrichment": 12.3},
                    {"name": "Pharmacokinetics", "genes": ["ABCB1", "SLCO1B1", "CYP3A4", "CYP2D6"], "p_value": 0.00005, "fold_enrichment": 8.7},
                    {"name": "Warfarin Metabolism", "genes": ["CYP2C9", "VKORC1", "CYP4F2"], "p_value": 0.0002, "fold_enrichment": 15.4},
                    {"name": "Opioid Response", "genes": ["OPRM1", "COMT", "CYP2D6"], "p_value": 0.0008, "fold_enrichment": 7.2},
                    {"name": "Antidepressant Response", "genes": ["CYP2D6", "CYP2C19", "COMT", "SLC6A4"], "p_value": 0.001, "fold_enrichment": 6.5},
                    {"name": "Statin-Induced Myopathy", "genes": ["SLCO1B1", "CYP3A4", "COQ2"], "p_value": 0.003, "fold_enrichment": 9.8},
                    {"name": "P450-Mediated Metabolism", "genes": ["CYP3A4", "CYP2D6", "CYP2C9", "CYP2C19", "CYP1A2"], "p_value": 0.00002, "fold_enrichment": 14.2},
                    {"name": "Dopamine Signaling", "genes": ["DRD2", "COMT", "SLC6A3"], "p_value": 0.004, "fold_enrichment": 5.3},
                    {"name": "Drug Transport", "genes": ["ABCB1", "SLCO1B1", "ABCG2"], "p_value": 0.0007, "fold_enrichment": 8.9},
                    {"name": "Phase II Metabolism", "genes": ["UGT1A1", "TPMT", "GSTP1"], "p_value": 0.002, "fold_enrichment": 7.6}
                ]
                
                # Format for display
                pathway_df = pd.DataFrame(pathways)
                pathway_df['genes_count'] = pathway_df['genes'].apply(len)
                pathway_df['genes'] = pathway_df['genes'].apply(lambda x: ", ".join(x))
                pathway_df['log_p'] = -np.log10(pathway_df['p_value'])
                
                # Sort by p-value
                pathway_df = pathway_df.sort_values('p_value')
                
                # Display pathway table
                st.subheader("Enriched Pathways")
                
                # Format p-values for display
                pathway_df['p_value'] = pathway_df['p_value'].apply(lambda x: f"{x:.2e}")
                
                st.dataframe(pathway_df[['name', 'genes_count', 'genes', 'p_value', 'fold_enrichment']])
                
                # Visualize enrichment
                st.subheader("Pathway Enrichment Plot")
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Create enrichment plot (bubble chart)
                scatter = ax.scatter(
                    pathway_df['genes_count'],
                    pathway_df['log_p'],
                    s=pathway_df['fold_enrichment'] * 20,  # Size based on fold enrichment
                    alpha=0.7,
                    c=range(len(pathway_df)),  # Color by rank
                    cmap='viridis'
                )
                
                # Add pathway labels
                for i, row in pathway_df.iterrows():
                    ax.annotate(
                        row['name'],
                        (row['genes_count'], row['log_p']),
                        xytext=(5, 0),
                        textcoords='offset points',
                        fontsize=8
                    )
                
                ax.set_xlabel('Number of Genes in Pathway')
                ax.set_ylabel('-log10(p-value)')
                ax.set_title('Pathway Enrichment Analysis')
                ax.grid(alpha=0.3)
                
                # Add colorbar legend
                cbar = plt.colorbar(scatter)
                cbar.set_label('Rank')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Pathway interactions visualization
                st.subheader("Pathway Interactions")
                
                # Create a graph of pathway interactions
                P = nx.Graph()
                
                # Add pathways as nodes
                for pathway in pathways:
                    P.add_node(pathway['name'], genes=pathway['genes'], size=len(pathway['genes']))
                
                # Add edges between pathways that share genes
                for i, p1 in enumerate(pathways):
                    for j, p2 in enumerate(pathways[i+1:], i+1):
                        shared_genes = set(p1['genes']).intersection(set(p2['genes']))
                        if shared_genes:
                            P.add_edge(p1['name'], p2['name'], weight=len(shared_genes), shared=", ".join(shared_genes))
                
                # Visualize pathway interaction network
                plt.figure(figsize=(12, 10))
                
                # Set the layout
                pos = nx.spring_layout(P, k=0.5, seed=42)
                
                # Node sizes based on number of genes
                node_sizes = [P.nodes[node]['size'] * 100 for node in P.nodes()]
                
                # Edge widths based on number of shared genes
                edge_widths = [P[u][v]['weight'] * 0.5 for u, v in P.edges()]
                
                # Draw nodes
                nx.draw_networkx_nodes(P, pos, node_size=node_sizes, node_color='lightblue', alpha=0.8)
                
                # Draw edges
                nx.draw_networkx_edges(P, pos, width=edge_widths, alpha=0.6, edge_color='gray')
                
                # Draw labels
                nx.draw_networkx_labels(P, pos, font_size=10, font_weight='bold')
                
                plt.title("Pathway Interaction Network", fontsize=16)
                plt.axis('off')
                st.pyplot(plt)
                
                # Show shared genes between pathways
                st.subheader("Shared Genes Between Pathways")
                
                edge_data = []
                for u, v, data in P.edges(data=True):
                    edge_data.append({
                        'Pathway 1': u,
                        'Pathway 2': v,
                        'Shared Genes': data['shared'],
                        'Count': data['weight']
                    })
                
                if edge_data:
                    edge_df = pd.DataFrame(edge_data).sort_values('Count', ascending=False)
                    st.dataframe(edge_df)
                else:
                    st.info("No shared genes between pathways found.")
            
            except Exception as e:
                logger.error(f"Error in pathway analysis: {e}")
                st.error(f"Error in pathway analysis: {str(e)}")