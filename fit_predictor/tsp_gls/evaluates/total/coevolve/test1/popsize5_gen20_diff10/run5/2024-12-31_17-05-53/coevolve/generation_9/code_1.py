import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the Manhattan distance for each edge
    manhattan_distance = np.abs(np.diff(distance_matrix, axis=0)).sum(axis=1)
    
    # Calculate the average edge distance for each node
    average_edge_distance = np.mean(distance_matrix, axis=1)
    
    # Create the heuristic matrix by combining the Manhattan distance and average edge distance
    heuristic_matrix = manhattan_distance + average_edge_distance
    
    # Normalize the heuristic matrix for better edge selection
    max_heuristic = np.max(heuristic_matrix)
    min_heuristic = np.min(heuristic_matrix)
    normalized_heuristic_matrix = (heuristic_matrix - min_heuristic) / (max_heuristic - min_heuristic)
    
    return normalized_heuristic_matrix