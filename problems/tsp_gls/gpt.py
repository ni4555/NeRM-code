import numpy as np
def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the recency weight (e.g., inverse of the number of times an edge has been visited)
    recency_weight = 1.0 / np.sum(distance_matrix, axis=0)
    
    # Calculate the distance weight (e.g., the actual distance of the edge)
    distance_weight = distance_matrix
    
    # Apply exponential decay to the recency weight
    decay_factor = 0.9
    recency_weight = np.exp(-decay_factor * recency_weight)
    
    # Combine the recency and distance weights
    combined_weight = recency_weight * distance_weight
    
    # Normalize the combined weights to ensure they sum to 1
    total_weight = np.sum(combined_weight)
    heuristics = combined_weight / total_weight
    
    return heuristics
