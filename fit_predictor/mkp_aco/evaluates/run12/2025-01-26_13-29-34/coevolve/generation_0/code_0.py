import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Initialize heuristic values with random values between 0 and 1
    heuristics = np.random.rand(prize.shape[0])
    
    # Normalize by dividing each item's prize by its total weight to get relative value
    relative_value = prize / (weight.sum(axis=1, keepdims=True))
    
    # Scale the relative values by the maximum relative value and add random noise
    scaled_heuristics = relative_value * relative_value.max() + np.random.normal scale=0.01, size=prize.shape[0]
    
    return scaled_heuristics
