import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Ensure the weights are 1D for each item, given the constraint of each dimension is 1
    if weight.ndim > 1:
        weight = weight.ravel()
    
    # Calculate the value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight
    
    # Initialize the heuristics array with the value-to-weight ratio
    heuristics = value_to_weight_ratio
    
    # Here you would implement the adaptive stochastic sampling and exploration strategy.
    # For the sake of this example, we'll just return the heuristics as calculated.
    # Note: This is where the dynamic ranking system and advanced exploration strategy would be implemented.
    
    return heuristics