"""
Visualization utilities for the Pharmacy Data Science Platform.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import networkx as nx

def set_plot_style():
    """Set consistent style for all plots"""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("notebook", font_scale=1.1)

def create_bar_chart(df, x_col, y_col, title=None, xlabel=None, ylabel=None, figsize=(10, 6)):
    """
    Create a bar chart from DataFrame columns
    
    Parameters:
    -----------
    df : DataFrame
        Data source
    x_col : str
        Column name for x-axis
    y_col : str
        Column name for y-axis
    title, xlabel, ylabel : str
        Chart labels
    figsize : tuple
        Figure size as (width, height)
        
    Returns:
    --------
    fig, ax
        Matplotlib figure and axis objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(df[x_col], df[y_col])
    
    # Add data labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
               f'{height:.2f}', ha='center', va='bottom', rotation=0)
    
    # Set labels
    ax.set_xlabel(xlabel or x_col)
    ax.set_ylabel(ylabel or y_col)
    ax.set_title(title or f"{y_col} by {x_col}")
    
    # Rotate x-axis labels if there are many categories
    if len(df) > 5:
        plt.xticks(rotation=45, ha='right')
    
    fig.tight_layout()
    return fig, ax

def create_line_chart(df, x_col, y_col, title=None, xlabel=None, ylabel=None, figsize=(10, 6), 
                      add_rolling_avg=False, window=7):
    """
    Create a line chart from DataFrame columns
    
    Parameters:
    -----------
    df : DataFrame
        Data source
    x_col, y_col : str
        Column names for axes
    title, xlabel, ylabel : str
        Chart labels
    figsize : tuple
        Figure size as (width, height)
    add_rolling_avg : bool
        Whether to add a rolling average line
    window : int
        Window size for rolling average
        
    Returns:
    --------
    fig, ax
        Matplotlib figure and axis objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot main data
    ax.plot(df[x_col], df[y_col], 'b-', alpha=0.5, label='Data')
    
    # Add rolling average if requested
    if add_rolling_avg and len(df) > window:
        y_avg = df[y_col].rolling(window=window).mean()
        ax.plot(df[x_col], y_avg, 'r-', label=f'{window}-Point Moving Average')
    
    # Set labels
    ax.set_xlabel(xlabel or x_col)
    ax.set_ylabel(ylabel or y_col)
    ax.set_title(title or f"{y_col} over {x_col}")
    ax.legend()
    
    fig.tight_layout()
    return fig, ax

def create_pie_chart(df, label_col, value_col, title=None, figsize=(10, 8), top_n=None):
    """
    Create a pie chart from DataFrame columns
    
    Parameters:
    -----------
    df : DataFrame
        Data source
    label_col : str
        Column name for pie slice labels
    value_col : str
        Column name for pie slice values
    title : str
        Chart title
    figsize : tuple
        Figure size as (width, height)
    top_n : int or None
        If set, only show top N items by value
        
    Returns:
    --------
    fig, ax
        Matplotlib figure and axis objects
    """
    # Make a copy to avoid modifying original
    data = df.copy()
    
    # Limit to top N items if requested
    if top_n is not None and len(data) > top_n:
        data = data.nlargest(top_n, value_col)
        remaining = df[~df.index.isin(data.index)]
        if not remaining.empty:
            # Add an "Other" category with the sum of remaining values
            other_row = pd.DataFrame({
                label_col: ['Other'],
                value_col: [remaining[value_col].sum()]
            })
            data = pd.concat([data, other_row])
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.pie(data[value_col], labels=data[label_col], autopct='%1.1f%%',
          startangle=90, shadow=True)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    ax.set_title(title or f"Distribution of {value_col} by {label_col}")
    
    return fig, ax

def create_scatter_plot(df, x_col, y_col, title=None, xlabel=None, ylabel=None, figsize=(10, 6),
                        add_trend=False, hue=None):
    """
    Create a scatter plot from DataFrame columns
    
    Parameters:
    -----------
    df : DataFrame
        Data source
    x_col, y_col : str
        Column names for axes
    title, xlabel, ylabel : str
        Chart labels
    figsize : tuple
        Figure size as (width, height)
    add_trend : bool
        Whether to add a trend line
    hue : str or None
        Column name for point colors
        
    Returns:
    --------
    fig, ax
        Matplotlib figure and axis objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create scatter plot
    if hue is not None and hue in df.columns:
        scatter = sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue, ax=ax)
    else:
        ax.scatter(df[x_col], df[y_col], alpha=0.7)
    
    # Add trend line if requested
    if add_trend:
        z = np.polyfit(df[x_col], df[y_col], 1)
        p = np.poly1d(z)
        ax.plot(df[x_col], p(df[x_col]), "r--", alpha=0.8)
        
        # Add correlation coefficient
        corr = df[x_col].corr(df[y_col])
        ax.text(0.05, 0.95, f"Correlation: {corr:.2f}", transform=ax.transAxes,
              fontsize=12, verticalalignment='top')
    
    # Set labels
    ax.set_xlabel(xlabel or x_col)
    ax.set_ylabel(ylabel or y_col)
    ax.set_title(title or f"{y_col} vs {x_col}")
    
    fig.tight_layout()
    return fig, ax

def create_heatmap(df, title=None, figsize=(10, 8), annot=True, cmap='coolwarm'):
    """
    Create a heatmap from a correlation matrix
    
    Parameters:
    -----------
    df : DataFrame
        Data source (typically a correlation matrix)
    title : str
        Chart title
    figsize : tuple
        Figure size as (width, height)
    annot : bool
        Whether to annotate cells with values
    cmap : str
        Colormap name
        
    Returns:
    --------
    fig, ax
        Matplotlib figure and axis objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    heatmap = sns.heatmap(df, annot=annot, cmap=cmap, ax=ax)
    ax.set_title(title or "Correlation Matrix")
    
    return fig, ax

def create_network_graph(G, node_color_map=None, edge_color_map=None, 
                         node_size_map=None, title=None, figsize=(12, 10)):
    """
    Create a network graph visualization
    
    Parameters:
    -----------
    G : NetworkX Graph
        The graph to visualize
    node_color_map : dict or None
        Mapping of nodes to colors
    edge_color_map : dict or None
        Mapping of edges to colors
    node_size_map : dict or None
        Mapping of nodes to sizes
    title : str
        Chart title
    figsize : tuple
        Figure size as (width, height)
        
    Returns:
    --------
    fig, ax
        Matplotlib figure and axis objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Position nodes using spring layout
    pos = nx.spring_layout(G, seed=42)
    
    # Set default node attributes
    node_colors = [node_color_map.get(node, 'lightblue') if node_color_map else 'lightblue' 
                   for node in G.nodes()]
                   
    node_sizes = [node_size_map.get(node, 300) if node_size_map else 300 
                  for node in G.nodes()]
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                          node_size=node_sizes, alpha=0.8, ax=ax)
    
    # Set default edge attributes
    edge_colors = [edge_color_map.get(edge, 'gray') if edge_color_map else 'gray' 
                   for edge in G.edges()]
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, 
                          edge_color=edge_colors, ax=ax)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
    
    ax.set_title(title or "Network Graph")
    ax.axis('off')
    
    return fig, ax

def create_gauge_chart(value, min_val=0, max_val=100, threshold_ranges=None, 
                       title=None, figsize=(10, 6)):
    """
    Create a gauge chart for a single value
    
    Parameters:
    -----------
    value : float
        The value to display
    min_val, max_val : float
        Range of the gauge
    threshold_ranges : list of tuples
        List of (threshold, color) tuples for coloring the gauge
    title : str
        Chart title
    figsize : tuple
        Figure size as (width, height)
        
    Returns:
    --------
    fig, ax
        Matplotlib figure and axis objects
    """
    if threshold_ranges is None:
        threshold_ranges = [
            (0, 'green'),     # 0-20: green
            (20, 'lightgreen'),  # 20-40: light green
            (40, 'yellow'),   # 40-60: yellow
            (60, 'orange'),   # 60-80: orange
            (80, 'red')       # 80-100: red
        ]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create gauge chart background
    gauge_angles = np.linspace(0, 180, 100)
    gauge_radii = [0.8] * 100
    
    # Determine color zones
    colors = []
    for angle in gauge_angles:
        angle_value = min_val + (angle / 180) * (max_val - min_val)
        for threshold, color in sorted(threshold_ranges):
            if angle_value >= threshold:
                current_color = color
            else:
                break
        colors.append(current_color)
    
    # Convert to radians for plotting
    gauge_angles_rad = np.radians(gauge_angles)
    
    # Plot gauge background
    ax.bar(gauge_angles_rad, gauge_radii, width=np.radians(180/100), 
           color=colors, edgecolor='gray', alpha=0.7)
    
    # Calculate needle angle
    normalized_value = (value - min_val) / (max_val - min_val)
    needle_angle = np.radians(normalized_value * 180)
    
    # Add needle to show current value
    ax.plot([0, np.sin(needle_angle)], [0, np.cos(needle_angle)], 'k-', lw=2)
    ax.add_patch(plt.Circle((0, 0), 0.05, color='black'))
    
    # Set up plot appearance
    ax.set_ylim(0, 1)
    ax.set_xlim(-1, 1)
    
    # Add labels for thresholds
    for i, (threshold, _) in enumerate(sorted(threshold_ranges)):
        if i > 0:  # Skip the first threshold (usually 0)
            threshold_angle = np.radians((threshold - min_val) / (max_val - min_val) * 180)
            label_x = 0.85 * np.sin(threshold_angle)
            label_y = 0.85 * np.cos(threshold_angle)
            ax.text(label_x, label_y, str(threshold), ha='center', va='center', fontsize=8)
    
    # Add value text
    ax.text(0, -0.2, f'Value: {value:.0f}', ha='center', fontsize=12, fontweight='bold')
    ax.set_title(title or 'Gauge Chart')
    
    # Remove axes
    ax.set_axis_off()
    
    return fig, ax

