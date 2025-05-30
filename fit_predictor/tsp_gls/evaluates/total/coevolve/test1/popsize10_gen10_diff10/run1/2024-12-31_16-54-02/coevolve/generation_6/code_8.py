import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = np.zeros_like(distance_matrix)
    
    # Compute the shortest path between any two nodes without returning to the starting node
    for i in range(len(distance_matrix)):
        for j in range(len(distance_matrix[i])):
            if i != j:
                # Compute the shortest path excluding the starting node (i.e., find shortest path from i to j)
                # Here we are using a simple approach to calculate the shortest path between two nodes
                # which may not be optimal for the TSP but serves as an example of how to calculate heuristics.
                # In practice, a more sophisticated shortest path algorithm may be used.
                heuristic_matrix[i][j] = np.min(distance_matrix[i][j:] + distance_matrix[j][i:])
                
    return heuristic_matrix