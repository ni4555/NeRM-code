import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Normalize the prize values to ensure they sum to 1
    normalized_prize = prize / np.sum(prize)
    
    # The heuristic is the normalized prize value, since the weight for each item is 1
    heuristics = normalized_prize
    
    return heuristics

# Example usage:
# Let's assume we have 5 items with the following prize values:
prize_example = np.array([10, 20, 30, 40, 50])
# Since each item has a weight of 1 for each dimension, the weight array is simply:
weight_example = np.ones_like(prize_example)

# Calculate the heuristics for these items
heuristics_example = heuristics_v2(prize_example, weight_example)
print(heuristics_example)