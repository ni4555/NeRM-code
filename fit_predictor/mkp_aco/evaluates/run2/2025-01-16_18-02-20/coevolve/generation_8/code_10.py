import numpy as np
import numpy as np
import random

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight.sum(axis=1)
    
    # Normalize the value-to-weight ratio to ensure all items are on the same scale
    # This step ensures that items with extreme values do not dominate the ranking
    max_ratio = np.max(value_to_weight_ratio)
    normalized_ratio = value_to_weight_ratio / max_ratio
    
    # Apply a probabilistic selection mechanism
    # The probability of selecting an item is proportional to its normalized value-to-weight ratio
    probabilities = normalized_ratio / normalized_ratio.sum()
    
    # Initialize the heuristics array with the probabilities
    heuristics = probabilities
    
    # Adaptive sampling mechanism
    # We adjust the sampling probability based on the remaining space in the knapsacks
    # This can be a simple greedy approach where we increase the probability of items
    # that can be added without exceeding the knapsack's weight limit
    for i in range(len(prize)):
        remaining_space = 1 - weight[i].sum()
        if remaining_space > 0:
            # Increase the probability for items that can be added
            # This is a simple linear increase, but more complex adaptive mechanisms can be used
            probabilities[i] *= (1 + remaining_space / prize[i])
    
    # Normalize the probabilities again after the adaptive adjustment
    heuristics = probabilities / probabilities.sum()
    
    return heuristics