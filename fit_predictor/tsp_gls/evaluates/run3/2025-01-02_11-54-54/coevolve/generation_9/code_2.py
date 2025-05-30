import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Subtract the minimum distance from each row
    min_distance = np.min(distance_matrix, axis=1)
    normalized_distance = distance_matrix - min_distance[:, np.newaxis]
    
    # Correlate with the graph's total cost (for simplicity, we'll use the sum of all distances)
    total_cost = np.sum(distance_matrix)
    heuristic_values = normalized_distance / total_cost
    
    return heuristic_values