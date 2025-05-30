import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the prize-to-weight ratio for each item
    prize_to_weight_ratio = prize / weight
    
    # Normalize the ratio to ensure a proper heuristic value
    # This could be a simple min-max normalization or another method
    min_ratio = np.min(prize_to_weight_ratio)
    max_ratio = np.max(prize_to_weight_ratio)
    normalized_ratio = (prize_to_weight_ratio - min_ratio) / (max_ratio - min_ratio)
    
    # Sample the normalized ratio and select the top items based on the heuristic
    # Here we use a simple random sampling, but this could be replaced with more sophisticated methods
    num_items_to_select = np.sum(weight == 1)  # Assuming we want to select items that fit within the 1-dimensional constraints
    sampled_indices = np.random.choice(range(len(normalized_ratio)), num_items_to_select, replace=False)
    selected_heuristics = normalized_ratio[sampled_indices]
    
    # Return the selected heuristics as an array
    return selected_heuristics