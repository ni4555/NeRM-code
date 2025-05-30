import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n, m = prize.shape[0], weight.shape[1]
    assert prize.shape == weight.shape, "prize and weight arrays must have the same length"
    
    # Calculate the weighted ratio for each item
    weighted_ratio = prize / weight.sum(axis=1)
    
    # Initialize a random number generator for adaptive stochastic sampling
    rng = np.random.default_rng()
    
    # Initialize the heuristics array with the weighted ratio
    heuristics = np.copy(weighted_ratio)
    
    # Perform adaptive stochastic sampling to refine heuristics
    for _ in range(10):  # 10 iterations of adaptive sampling
        # Randomly select a subset of items
        subset_indices = rng.choice(n, size=int(n * 0.2), replace=False)
        subset_prize = prize[subset_indices]
        subset_weight = weight[subset_indices]
        
        # Update heuristics based on the selected subset
        heuristics[subset_indices] = heuristics[subset_indices] + \
                                     (subset_prize / subset_weight.sum(axis=1)) - \
                                     (prize / weight.sum(axis=1))
    
    # Normalize the heuristics to ensure they sum up to 1
    heuristics /= heuristics.sum()
    
    return heuristics