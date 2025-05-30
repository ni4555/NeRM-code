import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize an array with the same shape as the distance matrix to store heuristics
    heuristics = np.zeros_like(distance_matrix)
    
    # Calculate the minimum spanning tree (MST) using a heuristic approach
    # Here, we use the Prim's algorithm for simplicity, but other heuristics can be applied
    num_nodes = distance_matrix.shape[0]
    visited = np.zeros(num_nodes, dtype=bool)
    min_heap = [(0, 0)]  # (cost, node)
    total_cost = 0
    
    while len(min_heap) > 0:
        cost, node = min_heap.pop(0)
        if visited[node]:
            continue
        visited[node] = True
        total_cost += cost
        
        # Update the heuristics for the current node
        for neighbor in range(num_nodes):
            if neighbor != node and not visited[neighbor]:
                edge_cost = distance_matrix[node, neighbor]
                heuristics[node, neighbor] = edge_cost - total_cost
    
    return heuristics