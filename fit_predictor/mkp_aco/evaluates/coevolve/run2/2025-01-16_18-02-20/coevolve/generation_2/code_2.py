import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate normalized value for each item
    normalized_value = prize / np.sum(weight, axis=1)
    
    # Rank items by normalized value
    ranks = np.argsort(normalized_value)[::-1]
    
    # Initialize heuristic values to 0
    heuristics = np.zeros_like(prize)
    
    # Calculate heuristic for each item based on rank
    for rank, index in enumerate(ranks):
        # Dynamic weight adjustment: scale by rank
        adjusted_weight = weight[index] * (rank + 1)
        
        # Check if the adjusted weight is within constraints
        if np.all(adjusted_weight <= 1):
            heuristics[index] = rank + 1
    
    return heuristics