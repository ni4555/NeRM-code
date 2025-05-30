import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the Manhattan distance for each edge
    manhattan_distances = np.abs(np.diff(distance_matrix, axis=0)).sum(axis=1)
    
    # Calculate the average edge distance
    average_edge_distance = np.mean(manhattan_distances)
    
    # Create a heuristic matrix where each entry is the product of the edge's
    # Manhattan distance and its inverse relative to the average edge distance
    heuristic_matrix = manhattan_distances * (1 / average_edge_distance)
    
    # Normalize the heuristic matrix to ensure that all values are non-negative
    # and sum to 1 (probability distribution)
    heuristic_matrix = (heuristic_matrix - np.min(heuristic_matrix)) / np.max(heuristic_matrix)
    
    return heuristic_matrix