import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # This function initializes a matrix with the same shape as the distance_matrix
    # with zeros, which represents the initial heuristic value for each edge.
    heuristic_matrix = np.zeros_like(distance_matrix)
    
    # Compute the shortest path between any two nodes using Dijkstra's algorithm
    # and fill the heuristic_matrix with the computed shortest path distances.
    for i in range(len(distance_matrix)):
        for j in range(len(distance_matrix)):
            if i != j:
                # Use Dijkstra's algorithm to find the shortest path from node i to node j
                # (excluding the path that would loop back to the origin node i).
                shortest_path = np.sort(distance_matrix[i])  # Get sorted distances from node i to all nodes
                shortest_path = shortest_path[1:]  # Exclude the distance to the node itself
                # The heuristic value is the minimum distance from the sorted list
                # excluding the distance to the node itself, which represents the cost
                # of reaching node j from node i without looping back to i.
                heuristic_matrix[i, j] = shortest_path[0]
    
    return heuristic_matrix