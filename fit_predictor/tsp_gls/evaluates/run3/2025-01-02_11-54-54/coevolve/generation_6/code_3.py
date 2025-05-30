import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Subtract the minimum distance from each row to normalize distances
    min_distances = np.min(distance_matrix, axis=1, keepdims=True)
    normalized_distances = distance_matrix - min_distances
    
    # Calculate the cost of the graph as the sum of all row sums
    graph_cost = np.sum(normalized_distances, axis=1)
    
    # Create a heuristic matrix where each element is the normalized cost
    # divided by the graph cost for the corresponding row, multiplied by the
    # inverse of the overall graph cost.
    heuristic_matrix = normalized_distances / graph_cost
    heuristic_matrix *= 1 / np.sum(graph_cost)
    
    return heuristic_matrix