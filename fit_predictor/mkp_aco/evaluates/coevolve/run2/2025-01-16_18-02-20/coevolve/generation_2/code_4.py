import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Normalize the prize to get the value per unit weight for each item
    value_per_weight = prize / weight.sum(axis=1, keepdims=True)
    
    # Calculate the normalized value by summing across all dimensions
    normalized_value = value_per_weight.sum(axis=1)
    
    # Rank items based on normalized value in descending order
    ranked_items = np.argsort(-normalized_value)
    
    # Initialize the heuristics array with zeros
    heuristics = np.zeros_like(prize)
    
    # Calculate the cumulative weight of selected items
    cumulative_weight = np.zeros_like(weight)
    
    # Iterate over the ranked items
    for item in ranked_items:
        # Check if adding the current item does not exceed the weight constraint
        if np.all(cumulative_weight + weight[item] <= 1):
            # Increment the cumulative weight
            cumulative_weight += weight[item]
            # Set the heuristics value for the current item
            heuristics[item] = 1
    
    return heuristics