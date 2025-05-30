import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate weighted ratio for each item
    weighted_ratio = prize / weight.sum(axis=1)
    
    # Normalize weighted ratio by the maximum ratio to ensure all values are positive
    max_ratio = np.max(weighted_ratio)
    normalized_ratio = weighted_ratio / max_ratio
    
    # Calculate heuristic score as a combination of normalized ratio and prize
    # The heuristic score is a weighted sum where the weights are chosen to give more
    # importance to the normalized ratio (since we want to maximize the prize per unit weight)
    # and less importance to the prize itself (since we are already prioritizing by normalized ratio)
    heuristic_score = normalized_ratio * 0.8 + prize * 0.2
    
    # Sort items based on the heuristic score in descending order
    sorted_indices = np.argsort(heuristic_score)[::-1]
    
    # Create an array to store the heuristic values for each item
    heuristics = np.zeros_like(prize)
    
    # Sample intelligently to maximize prize accumulation while respecting the weight constraints
    # Here we assume a simple strategy of sampling the top items, but this can be replaced
    # with a more complex sampling mechanism if needed
    for index in sorted_indices:
        # Check if adding the current item respects the weight constraints
        if np.all(weight[index] <= 1):
            heuristics[index] = 1  # Mark as promising to include
            # Update the weight constraints (since each dimension's constraint is 1)
            weight = np.maximum(weight - weight[index], 0)
    
    return heuristics