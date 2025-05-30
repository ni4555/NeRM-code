import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the weighted ratio for each item
    weighted_ratio = prize / weight.sum(axis=1)
    
    # Normalize the prize by cumulative sum to handle items with varying total prize
    cumulative_prize = np.cumsum(prize)
    normalized_prize = cumulative_prize / cumulative_prize[-1]
    
    # Combine the heuristic using both weighted ratio and normalized prize
    combined_heuristic = weighted_ratio * normalized_prize
    
    # Return the heuristics array
    return combined_heuristic