import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight.sum(axis=1)
    
    # Normalize the value-to-weight ratio to ensure it is comparable across items
    max_ratio = np.max(value_to_weight_ratio)
    normalized_ratio = value_to_weight_ratio / max_ratio
    
    # Apply a stochastic sampling strategy to select the most promising items
    # This can be implemented in various ways; here, we simply use random selection
    # with a probability proportional to the normalized ratio
    # Note: In a real-world scenario, this part would be more complex and adaptive
    # to the evolving constraints and should use techniques like simulated annealing,
    # genetic algorithms, or other probabilistic optimization methods.
    random_state = np.random.default_rng()
    selection_probability = normalized_ratio / np.sum(normalized_ratio)
    selected_indices = random_state.choice(range(len(value_to_weight_ratio)), 
                                          size=len(value_to_weight_ratio), 
                                          replace=False, 
                                          p=selection_probability)
    
    # Return the heuristics as an array of selected indices
    return selected_indices