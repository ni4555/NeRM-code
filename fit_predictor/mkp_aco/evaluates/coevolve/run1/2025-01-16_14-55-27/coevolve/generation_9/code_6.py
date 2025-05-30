import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n = prize.shape[0]
    m = weight.shape[1]
    
    # Calculate the weighted ratio for each item
    weighted_ratio = prize / weight.sum(axis=1)
    
    # Sort items based on the weighted ratio in descending order
    sorted_indices = np.argsort(weighted_ratio)[::-1]
    
    # Initialize heuristics array
    heuristics = np.zeros(n)
    
    # Adaptive dynamic sorting
    for i in range(n):
        # Sample a subset of the remaining items
        if n - i <= 5:  # Use a fixed threshold for small subsets
            sample_size = n - i
        else:
            sample_size = max(3, int(n * 0.1))  # Use a percentage threshold for larger subsets
        
        # Get the indices of the sample
        sample_indices = sorted_indices[i:i+sample_size]
        
        # Update the heuristics for the current item
        for j in sample_indices:
            if j < i:
                heuristics[i] += heuristics[j]
        
        # Apply greedy heuristic: prefer items with higher weighted ratio
        heuristics[i] = heuristics[i] * weighted_ratio[i]
    
    return heuristics