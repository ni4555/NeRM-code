import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Normalize the prize by the sum of weights for each item to get the value per unit weight
    normalized_value = prize / np.sum(weight, axis=1, keepdims=True)
    
    # Calculate the total normalized value for each item
    total_normalized_value = np.sum(normalized_value, axis=1)
    
    # Rank items by total normalized value
    ranking = np.argsort(-total_normalized_value)
    
    # Calculate the normalized value for each dimension
    normalized_weight = weight / np.sum(weight, axis=1, keepdims=True)
    
    # Compute the heuristic score for each item
    heuristic = np.prod(normalized_weight, axis=1)
    
    # Adjust the heuristic based on the ranking
    heuristic *= ranking
    
    return heuristic