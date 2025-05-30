import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the total weight for each dimension
    total_weight = np.sum(weight, axis=1)
    
    # Normalize the weights to the range [0, 1]
    normalized_weight = weight / total_weight[:, np.newaxis]
    
    # Calculate the heuristic value as a product of prize and normalized weight
    heuristic = prize * normalized_weight
    
    # Enhance prize maximization by increasing the heuristic for higher prize items
    # and adaptive heuristic sampling by reducing the heuristic for items with
    # high weight ratio (weight/dimension) that are close to their capacity
    enhanced_heuristic = heuristic * (1 - np.minimum(1, heuristic / np.maximum(1e-5, total_weight)))
    
    return enhanced_heuristic
