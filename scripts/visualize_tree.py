#!/usr/bin/env python
"""
Script to visualize hierarchical tree structures from pickle files.
This script can be used to visualize the hierarchical structure of datasets 
like ImageNet or iNaturalist that are used in the project.

Usage:
    python visualize_tree.py [pickle_file] [output_file] [--max_depth MAX_DEPTH] [--width WIDTH] [--height HEIGHT]

Example:
    python visualize_tree.py ../data/tiered_imagenet_tree.pkl tree_visualization.png --max_depth 4
"""

import argparse
import pickle
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import networkx as nx
from nltk.tree import Tree
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize tree structure from a pickle file')
    parser.add_argument('pickle_file', type=str, help='Path to the pickle file containing the tree')
    parser.add_argument('output_file', type=str, help='Path to save the visualization')
    parser.add_argument('--max_depth', type=int, default=3, help='Maximum depth of the tree to visualize')
    parser.add_argument('--width', type=float, default=30, help='Width of the figure in inches')
    parser.add_argument('--height', type=float, default=20, help='Height of the figure in inches')
    parser.add_argument('--interactive', action='store_true', help='Create interactive visualization')
    return parser.parse_args()

def load_tree(pickle_file):
    """Load a tree from a pickle file"""
    print(f"Loading tree from {pickle_file}")
    with open(pickle_file, 'rb') as f:
        tree = pickle.load(f)
    return tree

def create_networkx_graph(tree, max_depth=None):
    """Convert NLTK tree to NetworkX graph for visualization"""
    G = nx.DiGraph()
    
    def process_node(node, parent_id=None, depth=0):
        if max_depth is not None and depth > max_depth:
            return
        
        # Create unique ID for this node
        current_id = f"{id(node)}"
        
        # Add node with label
        if isinstance(node, Tree):
            label = node.label()
            G.add_node(current_id, label=str(label), is_leaf=False, depth=depth)
            
            # Connect to parent if this isn't the root
            if parent_id:
                G.add_edge(parent_id, current_id)
            
            # Process children
            for child in node:
                process_node(child, current_id, depth + 1)
        else:
            # Leaf node
            label = node
            G.add_node(current_id, label=str(label), is_leaf=True, depth=depth)
            if parent_id:
                G.add_edge(parent_id, current_id)
    
    process_node(tree)
    return G

def adjust_positions_for_pedigree(pos):
    """Adjust node positions to ensure proper pedigree-like layout"""
    # Flip y-coordinates to ensure root is at top
    return {node: (x, -y) for node, (x, y) in pos.items()}


def visualize_tree(tree, output_file, max_depth=3, width=20, height=12, interactive=False):
    """Create a visualization of the tree"""
    global nx
    G = create_networkx_graph(tree, max_depth=max_depth)
    
    # Print some statistics
    print(f"Tree statistics:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  Leaf nodes: {sum(1 for _, attrs in G.nodes(data=True) if attrs.get('is_leaf', False))}")
    
    # Calculate positions with hierarchical layout
    if interactive:
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
    else:
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
    # Set up the figure
    plt.figure(figsize=(width, height))
    
    # Draw nodes
    leaf_nodes = [node for node, attrs in G.nodes(data=True) if attrs.get('is_leaf', False)]
    internal_nodes = [node for node, attrs in G.nodes(data=True) if not attrs.get('is_leaf', False)]
    
    # Draw edges
    # Draw edges with straight lines
    nx.draw_networkx_edges(G, pos, alpha=0.5, arrows=False)
    
    # Draw nodes with different colors for leaf vs internal
    nx.draw_networkx_nodes(G, pos, nodelist=leaf_nodes, node_size=50, node_color='green', alpha=0.8)
    nx.draw_networkx_nodes(G, pos, nodelist=internal_nodes, node_size=100, node_color='blue', alpha=0.6)
    
    # Add labels for important nodes (e.g., first few levels)
    important_nodes = [node for node, attrs in G.nodes(data=True) 
                       if attrs.get('depth', float('inf')) < max(2, max_depth-1)]
    labels = {node: G.nodes[node]['label'] for node in important_nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
    
    # Create legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Internal Node'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Leaf Node')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Set title
    plt.title(f"Tree Visualization (max depth: {max_depth})")
    plt.axis('off')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_file}")
    
    if interactive:
        # Also save an interactive HTML version using Plotly
        try:
            import plotly.graph_objects as go
            import networkx as nx
            
            # Create edges for Plotly
            edge_x = []
            edge_y = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                
            # Create edge trace
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines')
            
            # Create nodes for Plotly
            node_x = []
            node_y = []
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                
            # Create node trace
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                hoverinfo='text',
                marker=dict(
                    showscale=True,
                    colorscale='YlGnBu',
                    size=10,
                    colorbar=dict(
                        thickness=15,
                        title='Node Connections',
                        xanchor='left'
                    )
                )
            )
            
            # Set node attributes for hover text
            node_adjacencies = []
            node_text = []
            for node, adjacencies in enumerate(G.adjacency()):
                node_adjacencies.append(len(adjacencies[1]))
                node_info = G.nodes[adjacencies[0]]
                node_text.append(f"Label: {node_info['label']}<br>Depth: {node_info['depth']}")
                
            # Update traces
            node_trace.marker.color = node_adjacencies
            node_trace.text = node_text
            
            # Create figure
            fig = go.Figure(data=[edge_trace, node_trace],
                         layout=go.Layout(
                            title=f'Tree Visualization (Interactive)',
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                            )
            
            # Save interactive HTML
            html_output = output_file.rsplit('.', 1)[0] + '.html'
            fig.write_html(html_output)
            print(f"Interactive visualization saved to {html_output}")
        except ImportError:
            print("Plotly not installed. Skipping interactive visualization.")
    
    return G

def main():
    args = parse_args()
    
    # Load tree from pickle file
    tree = load_tree(args.pickle_file)
    
    # Visualize tree
    visualize_tree(
        tree, 
        args.output_file, 
        max_depth=args.max_depth, 
        width=args.width, 
        height=args.height,
        interactive=args.interactive
    )

if __name__ == "__main__":
    main()