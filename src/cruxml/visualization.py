"""
Visualization utilities for CruxML

This module provides functions to visualize computational graphs and training progress.
"""

try:
    from graphviz import Digraph
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False
    print("Warning: graphviz not available. Graph visualization will not work.")

import matplotlib.pyplot as plt


def trace(root):
    """
    Build a set of all nodes and edges in the computational graph.
    
    Args:
        root: The root Value node to trace from
        
    Returns:
        Tuple of (nodes, edges) sets
    """
    nodes, edges = set(), set()
    
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    
    build(root)
    return nodes, edges


def draw_graph(root, filename=None):
    """
    Draw the computational graph using Graphviz.
    
    Args:
        root: The root Value node to visualize
        filename: Optional filename to save the graph
        
    Returns:
        Graphviz Digraph object
    """
    if not GRAPHVIZ_AVAILABLE:
        print("Error: graphviz is not available. Cannot draw graph.")
        return None
    
    dot = Digraph(format="svg", graph_attr={"rankdir": "LR"})
    
    nodes, edges = trace(root)
    
    for n in nodes:
        uid = str(id(n))
        # Create a rectangular node for each value
        dot.node(
            name=uid, 
            label="{ %s | data %.4f | grad %.4f}" % (n.label, n.data, n.grad), 
            shape='record'
        )
        if n._op:
            # Create an operation node
            dot.node(name=uid + n._op, label=n._op)
            # Connect operation to its result
            dot.edge(uid + n._op, uid)
    
    for n1, n2 in edges:
        # Connect operands to operation
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    
    if filename:
        dot.render(filename, cleanup=True)
    
    return dot


def plot_training_history(history, title="Training Loss"):
    """
    Plot training loss over epochs.
    
    Args:
        history: Training history dictionary with 'loss' key
        title: Title for the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'])
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()


def plot_predictions(X, y_true, y_pred, title="Predictions vs True Values"):
    """
    Plot predictions against true values for 1D problems.
    
    Args:
        X: Input data (for x-axis)
        y_true: True target values
        y_pred: Predicted values
        title: Title for the plot
    """
    plt.figure(figsize=(10, 6))
    
    if isinstance(X[0], list) and len(X[0]) == 1:
        # Extract single feature for plotting
        x_vals = [x[0] for x in X]
    else:
        # Use indices if multi-dimensional
        x_vals = range(len(X))
    
    # Extract values from Value objects if needed
    if hasattr(y_pred[0], 'data'):
        y_pred_vals = [y.data for y in y_pred]
    else:
        y_pred_vals = y_pred
    
    plt.scatter(x_vals, y_true, label='True Values', alpha=0.7)
    plt.scatter(x_vals, y_pred_vals, label='Predictions', alpha=0.7)
    plt.title(title)
    plt.xlabel('Input' if isinstance(X[0], list) and len(X[0]) == 1 else 'Sample Index')
    plt.ylabel('Output')
    plt.legend()
    plt.grid(True)
    plt.show()
