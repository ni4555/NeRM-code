import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the prize-to-weight ratio for each item
    ratio = prize / weight.sum(axis=1, keepdims=True)
    
    # Calculate the cumulative ratio to account for the multi-dimensional constraints
    cumulative_ratio = np.cumsum(ratio, axis=1)
    
    # Apply a multi-criteria ranking system based on the cumulative ratio
    # Here, we use a simple approach by multiplying by a constant to scale the values
    # This is a placeholder for a more complex ranking system if needed
    rank = cumulative_ratio * 1000
    
    # Normalize the rank to get the heuristic scores
    max_rank = np.max(rank, axis=1, keepdims=True)
    heuristics = rank / max_rank
    
    return heuristics