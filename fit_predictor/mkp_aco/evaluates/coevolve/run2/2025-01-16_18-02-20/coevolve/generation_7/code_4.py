import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight
    
    # Normalize the ratios to ensure they are comparable and prioritize items with higher ratios
    max_ratio = np.max(value_to_weight_ratio)
    normalized_ratios = value_to_weight_ratio / max_ratio
    
    # Use a simple stochastic sampling technique to select items based on the normalized ratios
    # Here we're using a random choice, but this could be replaced with a more sophisticated method
    # like Thompson sampling or other probabilistic selection techniques.
    random_choice = np.random.rand(len(normalized_ratios))
    selected_indices = np.argsort(random_choice)[:len(normalized_ratios)]
    
    # Return the selected indices as heuristics, which represent the order of item selection
    heuristics = np.zeros_like(prize)
    heuristics[selected_indices] = 1
    
    return heuristics