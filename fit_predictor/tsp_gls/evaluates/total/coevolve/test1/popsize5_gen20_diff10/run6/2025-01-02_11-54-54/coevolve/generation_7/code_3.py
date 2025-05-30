import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize a matrix of the same shape as the distance_matrix with zeros
    heuristics_matrix = np.zeros_like(distance_matrix)
    
    # Calculate the total graph cost (sum of all edges)
    total_graph_cost = np.sum(distance_matrix)
    
    # Calculate the minimum distance per node (sum of distances from each node to all other nodes)
    min_distances_per_node = np.sum(distance_matrix, axis=1)
    
    # Compute edge costs relative to the total graph cost
    edge_costs = distance_matrix / total_graph_cost
    
    # Adjust heuristics based on minimum distances per node
    heuristics_matrix = edge_costs - min_distances_per_node / total_graph_cost
    
    return heuristics_matrix