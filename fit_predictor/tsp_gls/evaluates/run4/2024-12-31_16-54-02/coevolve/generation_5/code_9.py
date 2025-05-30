import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize an array of the same shape as the distance matrix to store the heuristics
    heuristics = np.zeros_like(distance_matrix)
    
    # The edge distance heuristic is a dynamic shortest path algorithm.
    # For simplicity, we'll use Dijkstra's algorithm for each node as our dynamic shortest path algorithm.
    # In practice, this could be optimized using a priority queue to avoid recomputation.
    for i in range(len(distance_matrix)):
        # Create a copy of the distance matrix for Dijkstra's algorithm to operate on
        unvisited = distance_matrix[i].copy()
        unvisited[i] = np.inf
        shortest_path_tree = {i: 0}  # Initialize the tree with the starting node
        previous_nodes = {i: None}
        
        while shortest_path_tree:
            # Select the node with the smallest distance from the unvisited set
            current_node = min(shortest_path_tree, key=lambda k: unvisited[k])
            
            # Remove the current node from the unvisited set
            del unvisited[current_node]
            del shortest_path_tree[current_node]
            
            # Check the neighbors of the current node
            for neighbor, distance in enumerate(distance_matrix[current_node]):
                if distance != 0 and neighbor not in shortest_path_tree:
                    # Update the shortest path tree with the new distance
                    shortest_path_tree[neighbor] = distance
                    previous_nodes[neighbor] = current_node
                    # Set the heuristic value for the edge to the current node's shortest path distance
                    heuristics[i, neighbor] = distance
    
    return heuristics