import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight
    
    # Normalize the ratios to get a probability of selection
    normalized_ratio = value_to_weight_ratio / np.sum(value_to_weight_ratio)
    
    # Use adaptive stochastic sampling to sample items based on the normalized ratio
    heuristics = np.random.choice(np.arange(prize.shape[0]), size=prize.shape[0], replace=True, p=normalized_ratio)
    
    # Select items iteratively to maximize the prize without exceeding the weight constraints
    selected_items = []
    total_weight = 0
    for item in heuristics:
        if total_weight + weight[item] <= 1:  # Assuming the weight constraint is fixed to 1
            selected_items.append(item)
            total_weight += weight[item]
    
    # Return the heuristics for the selected items
    return np.array(selected_items)