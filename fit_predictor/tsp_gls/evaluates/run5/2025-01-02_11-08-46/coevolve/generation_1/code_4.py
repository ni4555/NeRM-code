import numpy as np
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Create a graph from the distance matrix
    graph = minimum_spanning_tree(distance_matrix)
    
    # Get the edge weights from the MST
    edge_weights = graph.data
    
    # Create a matrix of the same shape as the distance matrix
    heuristics_matrix = np.zeros_like(distance_matrix)
    
    # Assign the MST edge weights to the corresponding edges in the heuristic matrix
    for i in range(len(edge_weights)):
        # The edge index in the MST corresponds to the row and column of the distance matrix
        edge_row, edge_col = graph.indices[i], graph.indptr[i+1] - 1
        heuristics_matrix[edge_row, edge_col] = edge_weights[i]
        heuristics_matrix[edge_col, edge_row] = edge_weights[i]
    
    return heuristics_matrix