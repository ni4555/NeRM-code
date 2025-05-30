import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the multi-dimensional weighted ratio metric for each item
    weighted_ratio = np.prod(weight, axis=1) / prize
    
    # Apply a dynamic sorting mechanism based on the weighted ratio
    # The sorting key is a tuple where the first element is the negative of the weighted ratio
    # (we use negative because we want the maximum ratio first) and the second element is the index
    # This ensures that the sorting is stable and index information is preserved
    sorted_indices = np.argsort((-weighted_ratio, np.arange(prize.size)))
    
    # Use cumulative sum analysis to assess item contribution
    # We assume the contribution of an item is inversely proportional to the weighted ratio
    cumulative_contribution = np.cumsum(1.0 / weighted_ratio)
    
    # Create a heuristics array where each element indicates how promising it is to include item i
    # We use a simple inverse proportional heuristic (i.e., the more promising an item is, the higher its score)
    heuristics = 1.0 / weighted_ratio[sorted_indices] * cumulative_contribution[sorted_indices]
    
    return heuristics