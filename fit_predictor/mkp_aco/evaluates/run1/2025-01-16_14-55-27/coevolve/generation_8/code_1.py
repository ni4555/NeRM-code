import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n = len(prize)
    m = len(weight[0])
    
    # Step 1: Calculate the total weight for each item
    total_weight = np.sum(weight, axis=1)
    
    # Step 2: Calculate the weighted ratio for each item
    weighted_ratio = prize / total_weight
    
    # Step 3: Dynamic multi-criteria sorting
    # Assuming we sort by the weighted ratio in descending order and then by prize in descending order
    # Note: This is a simplification; a more complex sorting mechanism could be implemented here
    sorted_indices = np.argsort(weighted_ratio)[::-1]
    sorted_indices = np.argsort(prize[sorted_indices])[::-1][sorted_indices]
    
    # Step 4: Apply a robust heuristic normalization technique
    # We normalize by the maximum weighted ratio
    max_weighted_ratio = np.max(weighted_ratio)
    normalized_weights = weighted_ratio / max_weighted_ratio
    
    # Step 5: Return the normalized weights as heuristics
    heuristics = normalized_weights[sorted_indices]
    
    return heuristics