import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight.sum(axis=1)
    
    # Normalize the ratios to make them comparable across items
    normalized_ratios = value_to_weight_ratio / value_to_weight_ratio.sum()
    
    # Create a binary vector indicating the probability of selection for each item
    selection_probability = np.random.rand(len(prize))
    
    # Sample items based on the normalized ratios
    item_indices = np.argsort(normalized_ratios)[::-1]
    selected_item_indices = item_indices[np.random.choice(len(item_indices), size=int(len(prize) / 2), replace=False)]
    
    # Initialize the heuristics array with zeros
    heuristics = np.zeros_like(prize)
    
    # Set the heuristics for selected items to 1
    heuristics[selected_item_indices] = 1
    
    return heuristics