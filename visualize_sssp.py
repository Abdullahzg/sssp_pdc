import json
import matplotlib.pyplot as plt
import networkx as nx
import argparse

def load_graph_from_json(filename):
    """Load graph from JSON file."""
    with open(filename, 'r') as f:
        data = json.load(f)
    
    G = nx.DiGraph()
    
    # Add nodes with attributes
    for node in data['nodes']:
        node_id = node['id']
        distance = node['distance']
        parent = node['parent']
        
        # Convert -1 distance to 'inf' for display
        if distance == -1:
            display_distance = 'inf'
        else:
            display_distance = f"{distance:.1f}"
        
        G.add_node(node_id, distance=distance, parent=parent, label=f"{node_id}\nDist: {display_distance}")
    
    # Add edges with attributes
    for edge in data['edges']:
        source = edge['source']
        target = edge['target']
        weight = edge['weight']
        is_tree_edge = edge['is_tree_edge']
        
        G.add_edge(source, target, weight=weight, is_tree_edge=is_tree_edge)
    
    return G, data['source']

def load_changes_from_json(filename):
    """Load edge changes from JSON file."""
    with open(filename, 'r') as f:
        data = json.load(f)
    
    return data['changes']

def visualize_sssp_tree(G, source, title, filename):
    """Visualize the SSSP tree."""
    plt.figure(figsize=(12, 10))
    
    # Create edge lists for tree edges and non-tree edges
    tree_edges = [(u, v) for u, v, d in G.edges(data=True) if d['is_tree_edge']]
    non_tree_edges = [(u, v) for u, v, d in G.edges(data=True) if not d['is_tree_edge']]
    
    # Get node positions using spring layout
    pos = nx.spring_layout(G, seed=42)  # Fixed seed for reproducibility
    
    # Draw nodes
    node_colors = []
    for node in G.nodes():
        if node == source:
            node_colors.append('red')  # Source node in red
        elif G.nodes[node]['distance'] == -1:
            node_colors.append('grey')  # Unreachable nodes in grey
        else:
            node_colors.append('skyblue')  # Regular nodes in blue
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edgelist=non_tree_edges, width=1.0, alpha=0.5, edge_color='grey', style='dashed')
    nx.draw_networkx_edges(G, pos, edgelist=tree_edges, width=2.0, alpha=1.0, edge_color='blue')
    
    # Draw edge weights
    edge_labels = {(u, v): f"{d['weight']:.1f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    # Draw node labels
    node_labels = {node: G.nodes[node]['label'] for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
    
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def visualize_changes(changes, filename):
    """Visualize the edge changes."""
    plt.figure(figsize=(10, 8))
    
    # Count operations
    insertions = [c for c in changes if c['operation'] == 'insert']
    deletions = [c for c in changes if c['operation'] == 'delete']
    
    plt.bar(['Insertions', 'Deletions'], [len(insertions), len(deletions)])
    plt.title(f'Edge Changes Summary: {len(changes)} total changes')
    plt.ylabel('Count')
    plt.savefig(filename, dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize SSSP trees before and after edge changes')
    parser.add_argument('--before', default='graph_before.json', help='Path to before graph JSON')
    parser.add_argument('--after', default='graph_after.json', help='Path to after graph JSON')
    parser.add_argument('--changes', default='edge_changes.json', help='Path to edge changes JSON')
    args = parser.parse_args()
    
    # Load graphs
    before_graph, source = load_graph_from_json(args.before)
    after_graph, _ = load_graph_from_json(args.after)
    
    # Load changes
    changes = load_changes_from_json(args.changes)
    
    # Count nodes and edges
    num_nodes = before_graph.number_of_nodes()
    num_edges = before_graph.number_of_edges()
    num_tree_edges_before = sum(1 for _, _, d in before_graph.edges(data=True) if d['is_tree_edge'])
    num_tree_edges_after = sum(1 for _, _, d in after_graph.edges(data=True) if d['is_tree_edge'])
    
    # Calculate affected vertices
    affected_distances = 0
    affected_parents = 0
    
    for node in before_graph.nodes():
        if before_graph.nodes[node]['distance'] != after_graph.nodes[node]['distance']:
            affected_distances += 1
        if before_graph.nodes[node]['parent'] != after_graph.nodes[node]['parent']:
            affected_parents += 1
    
    # Print statistics
    print("Graph Statistics:")
    print(f"  Number of nodes: {num_nodes}")
    print(f"  Number of edges: {num_edges}")
    print(f"  Number of tree edges before update: {num_tree_edges_before}")
    print(f"  Number of tree edges after update: {num_tree_edges_after}")
    print("\nEdge Changes:")
    print(f"  Total changes: {len(changes)}")
    print(f"  Insertions: {sum(1 for c in changes if c['operation'] == 'insert')}")
    print(f"  Deletions: {sum(1 for c in changes if c['operation'] == 'delete')}")
    print("\nSSP Update Effects:")
    print(f"  Vertices with changed distances: {affected_distances}")
    print(f"  Vertices with changed parents: {affected_parents}")
    
    # Generate visualizations
    visualize_sssp_tree(before_graph, source, "SSSP Tree Before Edge Changes", "sssp_before.png")
    visualize_sssp_tree(after_graph, source, "SSSP Tree After Edge Changes", "sssp_after.png")
    visualize_changes(changes, "edge_changes.png")
    
    print("\nVisualizations generated:")
    print("  SSSP tree before edge changes: sssp_before.png")
    print("  SSSP tree after edge changes: sssp_after.png")
    print("  Edge changes summary: edge_changes.png")

if __name__ == "__main__":
    main() 