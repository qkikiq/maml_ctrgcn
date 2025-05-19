import numpy as np
import matplotlib.pyplot as plt
from graph.ntu_rgb_d import Graph
import graph.tools as tools


def visualize_adjacency_matrices():
    # Create graph instance
    graph = Graph(labeling_mode='spatial')

    # Get the three components of the adjacency matrix
    num_node = 25
    self_matrix = np.zeros((num_node, num_node))
    inward_matrix = np.zeros((num_node, num_node))
    outward_matrix = np.zeros((num_node, num_node))

    # Fill matrices
    for i, j in graph.self_link:
        self_matrix[i][j] = 1
    for i, j in graph.inward:
        inward_matrix[i][j] = 1
    for i, j in graph.outward:
        outward_matrix[i][j] = 1

    # Create subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Plot matrices
    ax1.imshow(self_matrix, cmap='Blues')
    ax1.set_title('Self-Loop Matrix')
    ax1.set_xlabel('Node')
    ax1.set_ylabel('Node')

    ax2.imshow(inward_matrix, cmap='Blues')
    ax2.set_title('Inward Matrix')
    ax2.set_xlabel('Node')
    ax2.set_ylabel('Node')

    ax3.imshow(outward_matrix, cmap='Blues')
    ax3.set_title('Outward Matrix')
    ax3.set_xlabel('Node')
    ax3.set_ylabel('Node')

    plt.tight_layout()
    plt.show()


# Call the visualization function
visualize_adjacency_matrices()