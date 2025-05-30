import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n, m = prize.shape[0], weight.shape[1]
    # Ensure weight is 1 for all dimensions
    if weight.shape[1] != 1:
        raise ValueError("Expected weight to have a single dimension with value 1 for each item.")
    
    # Calculate the weighted ratio for each item
    weighted_ratio = prize / weight
    
    # Normalize the prize by the cumulative prize sum
    cumulative_prize = np.cumsum(prize)
    normalized_prize = prize / cumulative_prize
    
    # Combine the weighted ratio and normalized prize for heuristic
    combined_heuristics = weighted_ratio * normalized_prize
    
    # Return the resulting heuristics array
    return combined_heuristics

# Example usage:
# prize = np.array([10, 20, 30])
# weight = np.array([[1], [1], [1]])
# print(heuristics_v2(prize, weight))