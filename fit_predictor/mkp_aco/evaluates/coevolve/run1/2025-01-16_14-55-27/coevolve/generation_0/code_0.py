import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Assuming that the constraint for each dimension is fixed to 1,
    # we can use a simple heuristic that considers the ratio of prize to weight.
    # This heuristic assumes that each item's weight is in the same dimension.
    # We'll calculate the "prominence" of each item based on its prize-to-weight ratio.
    
    # Check if the weights have only one dimension
    if weight.ndim == 1:
        weight = weight[:, np.newaxis]  # Reshape to (n, 1) if it's not already
    
    # Calculate the prize-to-weight ratio for each item
    prize_to_weight_ratio = prize / weight.sum(axis=1)
    
    # Normalize the ratios to sum to 1, which will give us the "prominence" of each item
    prominence = prize_to_weight_ratio / prize_to_weight_ratio.sum()
    
    return prominence