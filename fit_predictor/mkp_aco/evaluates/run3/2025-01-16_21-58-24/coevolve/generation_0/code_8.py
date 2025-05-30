import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Initialize the heuristics array with zeros
    heuristics = np.zeros_like(prize, dtype=float)
    
    # Normalize the weight matrix to the range [0, 1]
    weight_normalized = weight / weight.sum(axis=1, keepdims=True)
    
    # Calculate the heuristics for each item
    for i in range(prize.shape[0]):
        # Compute the expected profit for item i, considering the normalized weights
        expected_profit = prize[i] * weight_normalized[i].sum()
        # Set the heuristics value as the expected profit
        heuristics[i] = expected_profit
    
    return heuristics